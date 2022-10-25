"""This module contains the main implementation of the ShapFire method for
feature ranking and selection."""

import logging
import typing

import lightgbm
import numpy
import pandas
import shap
import sklearn
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    is_classifier,
    is_regressor,
)
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
)
from tqdm import tqdm

import shapfire.utils as utils
from shapfire.clustering import (
    AutoHierarchicalAssociationClustering,
    ClusterSampler,
    _identify_colinear_features,
)
from shapfire.plotting import ShapFirePlottingInterface

DEFAULT_SPLITS: int = 2
"""The default number of folds a dataset should be divided into in a
cross-validation."""

DEFAULT_REPEATS: int = 1
"""The number of times, in a cross-validation, the division of a dataset into a
certain number of folds should be repeated.
"""

# Valid hyperparameter search methods
HYPERPARAMETER_SEARCH_METHODS: list[typing.Any] = [
    RandomizedSearchCV,
    GridSearchCV,
    None,
]

# Valid estimator classes
ESTIMATOR_CLASSES: list = [
    lightgbm.LGBMClassifier,
    lightgbm.LGBMRegressor,
    sklearn.ensemble.RandomForestClassifier,
    sklearn.ensemble.RandomForestRegressor,
]

# NOTE: All scorer objects follow the convention that higher return values are
# better than lower return values.
CLASSIFICATION_SCORING: dict[str, typing.Any] = {
    # "accuracy",
    # "balanced_accuracy",
    # "top_k_accuracy",
    # "average_precision",
    # "neg_brier_score",
    # "f1",
    # "f1_micro",
    # "f1_macro",
    # "f1_weighted",
    # "f1_samples",
    # "neg_log_loss",
    # "precision",
    # "precision_micro",
    # "precision_macro",
    # "precision_weighted",
    # "precision_samples",
    # "recall",
    # "recall_micro",
    # "recall_macro",
    # "recall_weighted",
    # "recall_samples",
    # "jaccard",
    # "jaccard_micro",
    # "jaccard_macro",
    # "jaccard_weighted",
    # "jaccard_samples",
    "roc_auc": {
        # Indicate that the score is by default positive
        "sign": 1,
        "best": 1.0,
        "worst": 0.0,
    },
    # "roc_auc_ovr",
    # "roc_auc_ovo",
    # "roc_auc_ovr_weighted",
    # "roc_auc_ovo_weighted",
}

# NOTE: All scorer objects follow the convention that higher return values are
# better than lower return values.
REGRESSION_SCORING: dict[str, typing.Any] = {
    "explained_variance": {
        # Indicate that the score is by default positive
        "sign": 1,
        "best": 1.0,
        "worst": -numpy.inf,
    },
    "max_error": {
        # Indicate that the score is by default negative
        "sign": -1,
        "best": 0.0,
        "worst": -numpy.inf,
    },
    "neg_mean_absolute_error": {
        # Indicate that the score is by default negative
        "sign": -1,
        "best": 0.0,
        "worst": -numpy.inf,
    },
    "neg_mean_squared_error": {
        # Indicate that the score is by default negative
        "sign": -1,
        "best": 0.0,
        "worst": -numpy.inf,
    }
    # "neg_root_mean_squared_error",
    # "neg_mean_squared_log_error",
    # "neg_median_absolute_error",
    # "r2",
    # "neg_mean_poisson_deviance",
    # "neg_mean_gamma_deviance",
    # "neg_mean_absolute_percentage_error",
}


def _check_estimator_class(
    estimator_class: typing.Union[
        lightgbm.LGBMClassifier,
        lightgbm.LGBMRegressor,
        sklearn.ensemble.RandomForestClassifier,
        sklearn.ensemble.RandomForestRegressor,
    ],
) -> None:
    if estimator_class not in ESTIMATOR_CLASSES:
        raise ValueError(
            f"The given estimator class {estimator_class} is not a "
            + "valid estimator."
        )


def _check_scoring_function(
    scoring: typing.Union[str, typing.Callable],
    estimator_class: typing.Union[
        lightgbm.LGBMClassifier,
        lightgbm.LGBMRegressor,
        sklearn.ensemble.RandomForestClassifier,
        sklearn.ensemble.RandomForestRegressor,
    ],
) -> None:
    # if 'self.scoring' is a string then make sure the specified
    # scoring function is lower-case and does not contain any whitespace
    # before checking whether it is actually a valid scoring function
    if isinstance(scoring, str):
        scoring = scoring.strip().lower()
        if is_classifier(estimator_class):
            if scoring not in list(CLASSIFICATION_SCORING.keys()):
                raise ValueError(
                    f"The given scoring function {scoring} is not a "
                    + "valid scorer for a classifiction task using "
                    + f"estimator {estimator_class}."
                )
        elif is_regressor(estimator_class):
            if scoring not in list(REGRESSION_SCORING.keys()):
                raise ValueError(
                    f"The given scoring function {scoring} is not a "
                    + "valid scorer for a regression task using estimator "
                    + f"{estimator_class}."
                )
        else:
            raise ValueError(
                "It could not be determined whether the given "
                + f"'ShapFire.estimator_class': {estimator_class} "
                + "is a classifier or a regressor."
            )
    elif isinstance(scoring, typing.Callable):  # type: ignore
        # TODO: Check that 'self.scoring' is a valid scoring function
        raise NotImplementedError(
            "It is currently not possible to pass a callable scoring "
            + "function"
        )


def _check_hyperparameter_search_params(
    hyperparameter_search: typing.Union[None, GridSearchCV, RandomizedSearchCV],
) -> None:
    if hyperparameter_search is not None:
        if (
            hyperparameter_search != GridSearchCV
            and hyperparameter_search != RandomizedSearchCV
        ):
            raise ValueError(
                "The given input argument 'hyperparameter_search': "
                + f"'{hyperparameter_search}' is not a valid "
                + "option. Valid input values are: "
                + ", ".join(HYPERPARAMETER_SEARCH_METHODS)
                + "."
            )


def _check_cv_params(n_splits: int, n_repeats: int) -> None:
    if n_splits < 2:
        raise ValueError(
            "The given input argument 'n_splits' can not be less " + "than 2."
        )
    if n_repeats < 1:
        raise ValueError(
            "The given input argument 'n_repeats' can not be" "less than 1."
        )


def _check_reference_vector_params(reference: str) -> None:
    if reference not in ["min", "max", "mean", "median"]:
        raise ValueError(
            f"The given input argument 'reference': {reference} "
            + " is not a  valid option. Valid options are: "
            + ", ".join(["min", "max", "median", "mean"])
            + "."
        )


def get_kfold_cross_validator(
    estimator_class: typing.Union[
        lightgbm.LGBMClassifier,
        lightgbm.LGBMRegressor,
        sklearn.ensemble.RandomForestClassifier,
        sklearn.ensemble.RandomForestRegressor,
    ],
    n_splits: int,
    n_repeats: int,
) -> typing.Union[RepeatedStratifiedKFold, RepeatedKFold]:
    """
    Based on the type of estimator that is used in the ShapFire method, this
    method determines how to divide a dataset into training and test folds.
    Calssifiers use repeated stratified k-fold cross-validation by default while
    regressors simply use repeated k-Fold cross-validation.

    Args:
        estimator_class: The scikit-learn or Microsoft LightGBM \
            tree-based estimator to use. The estimator can either be a \
            classifier or a regressor.
        n_splits: The number of folds a dataset should be divided into.
        n_repeats: The number of times the division of a dataset into a \
            certain number of folds should be repeated.

    Raises:
        ValueError: If it could not be determined whether the given \
            'estimator_class' is a classifier or a regressor."

    Returns:
        A scikit-learn cross-validator object that splits a given dataset into \
            training and test folds.
    """
    if is_classifier(estimator_class):
        return RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
        )
    elif is_regressor(estimator_class):
        return RepeatedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
        )
    else:
        raise ValueError(
            "It could not be determined whether the given "
            + f"'estimator_class': {estimator_class} "
            + "is a classifier or a regressor."
        )


def get_roc_auc_statistics(
    estimator: lightgbm.LGBMClassifier,
    X_test: typing.Union[numpy.ndarray, pandas.DataFrame],
    y_test: typing.Union[numpy.ndarray, pandas.DataFrame],
) -> tuple[numpy.ndarray, numpy.ndarray, float]:
    """
    For a binary classification task, compute the Area Under the Receiver
    Operating Characteristic Curve (ROC AUC). The ROC AUC score is calculated
    based on the prediction scores calculated by the given estimator.

    Args:
        estimator: A scikit-learn or Miscrosoft LightGBM estimator to use. \
            The estimator can either be a classifier or a regressor. The \
            estimator is assumed to have been trained on a training dataset \
            and should be evaluated on a test dataset.
        X_test: A test dataset.
        y_test: The samples associated with the target variable of the \
            test dataset.

    Raises:
        ValueError: If the given input argument 'estimator' is not a
            classifier.

    Returns:
        Return false positive rates (fpr), true positive rates (tpr), and
        the ROC AUC score.
    """
    # Make sure the estimator given as input argument is actually a classifier
    if is_classifier(estimator):
        # Retrieve 'fpr': False Positive Rate
        #          'tpr': True Positive Rate
        fpr, tpr, _ = roc_curve(
            y_true=y_test, y_score=estimator.predict_proba(X=X_test)[:, 1]
        )
        # Compute Area Under the Curve (AUC)
        roc_auc = auc(x=fpr, y=tpr)
        return fpr, tpr, roc_auc
    else:
        raise ValueError(
            "Internal Error. "
            + "The given input argument 'estimator' needs to be a "
            + "classifier."
        )


class HyperparameterSearchHelper(BaseEstimator):
    """
    A ShapFire helper class for performing cross-validation and hyperparameter
    tuning.

    Args:
        BaseEstimator: A scikit-learn estimator class used for API \
            compatibility purposes.
    """

    def __init__(
        self,
        cv: typing.Union[RepeatedStratifiedKFold, RepeatedKFold],
        estimator_class: typing.Union[
            lightgbm.LGBMClassifier,
            lightgbm.LGBMRegressor,
            sklearn.ensemble.RandomForestClassifier,
            sklearn.ensemble.RandomForestRegressor,
        ],
        estimator_params: typing.Union[None, dict[str, typing.Any]],
        scoring: str,
        hyperparameter_search: typing.Union[
            None, GridSearchCV, RandomizedSearchCV
        ] = None,
        n_jobs: typing.Union[None, int] = -1,
        random_seed: int = utils.DEFAULT_RANDOM_SEED,
    ) -> None:
        """
        A helper class to perform cross-validation and hyperparameter tuning,
        given (i) a valid way of generating cross-validation trin/test folds and
        (ii) a valid scikit-learn class for performing hyperparameter tuning.

        Args:
            cv: A scikit-learn cross-validator class for generating train/test \
                folds.
            estimator_class: The scikit-learn or Microsoft LightGBM \
                tree-based estimator to use. The estimator can either be a \
                classifier or a regressor.
            estimator_params: The estimator hyperparameters and corresponding \
                values to search or directly use. If only a single value for \
                each hyperparameter is provided then only cross-validation \
                will be performed and no hyperparameter search will be \
                performed. Defaults to None.
            scoring: The specification of a scoring function to use for \
                model-evaluation, i.e., a function that can be used for \
                assessing the prediction error of a trained model given a test \
                set.
            hyperparameter_search: The type of hyperparameter search method to \
                apply. Defaults to None, which simply results in the \
                cross-validation.
            n_jobs: The number of jobs to run in parallel. None means 1 while \
                -1 means use all processor cores. Defaults to -1.
            random_seed: The random seed to use for \
                reproducibility purposes. Defaults to \
                    :const:`shapfire.utils.DEFAULT_RANDOM_SEED`.
        """
        # Class variables corresponding to calss input arguments
        self.cv = cv
        self.estimator_class = estimator_class
        self.estimator_params = estimator_params
        self.scoring = scoring
        self.hyperparameter_search = hyperparameter_search
        self.n_jobs = n_jobs
        self.random_seed = random_seed

        # Check that the given input is valid
        self._check_vars()

        # Publically accessible variables associated with the model that
        # obtained the best performance score. These variables wil eventually be
        # set after a call to 'fit()'
        self.best_score_: typing.Union[None, float] = None
        self.best_params_: typing.Union[None, dict[str, typing.Any]] = None

    def fit(
        self,
        X: typing.Union[numpy.ndarray, pandas.DataFrame],
        y: typing.Union[numpy.ndarray, pandas.Series],
    ) -> "HyperparameterSearchHelper":
        """
        Perform cross-validation and hyperparameter tuning given an input
        dataset. In case a single hyperparameter configuration is given as input
        then only cross-validation will be performed.

        Args:
            X: An input dataset that should be utilized when performing \
                hyperparameter tuning coupled with cross-validation. The \
                dataset is assumed to contain features (columns) and \
                corresponding observations (rows).
            y: The samples associated with the target variable of the dataset.

        Raises:
            ValueError: If an invalid hyperparameter search method is specified.

        Returns:
            An updated 'HyperparameterSearchHelper' class object that has been \
            updated with the results obtained from hyperparameter tuning \
            coupled with cross-validation. In case a single hyperparameter \
            configuration is given as input then only cross-validation will be \
            performed.
        """
        args = {
            # TODO: Scoring function should be passed as argument
            "scoring": self.scoring,
            "n_jobs": self.n_jobs,
            "cv": self.cv,
        }
        # Perform no hyperparameter search just fit the estimator
        # with the given 'estimator_params'
        if self.hyperparameter_search is None:
            # Create model object with set random state and given input
            # parameters
            args["estimator"] = self.estimator_class(
                random_state=self.random_seed,
                **self.estimator_params,
            )
            cv_scores = cross_val_score(
                X=X,
                y=y,
                **args,
            )
            means = numpy.mean(cv_scores)
            stds = numpy.std(cv_scores, ddof=1)
            params = self.estimator_params
            self.best_score_ = means
            self.best_params_ = self.estimator_params
            logging.info("\nGrid scores on development set:")
            logging.info("%0.3f (+/-%0.03f) for %r" % (means, stds * 2, params))
            logging.info(f"Best score ({self.scoring}): {self.best_score_}")
        # Otherwise perform 'GridSearchCV' or 'RandomizedSearchCV'
        elif (
            self.hyperparameter_search == GridSearchCV
            or self.hyperparameter_search == RandomizedSearchCV
        ):
            # Create model object with set random state
            args["estimator"] = self.estimator_class(
                random_state=self.random_seed
            )
            if self.hyperparameter_search == GridSearchCV:
                kwargs = {
                    "param_grid": self.estimator_params,
                    # NOTE: Do NOT refit estimator here on all the available
                    # data.
                    "refit": False,
                    "return_train_score": False,
                }
                args.update(kwargs)
            elif self.hyperparameter_search == RandomizedSearchCV:
                kwargs = {
                    "param_distributions": self.estimator_params,
                    # NOTE: Do NOT refit estimator here on all the available
                    # data.
                    "refit": False,
                    "return_train_score": False,
                }
                args.update(kwargs)
            search = self.hyperparameter_search(**args)
            search.fit(X=X, y=y)
            means = search.cv_results_["mean_test_score"]
            stds = search.cv_results_["std_test_score"]
            params = search.cv_results_["params"]
            self.best_score_ = search.best_score_
            self.best_params_ = search.best_params_
            # Report back some results...
            logging.info("\nGrid scores on development set:")
            for _mean, _std, _params in zip(  # noqa: FKA01
                means, stds, [params]
            ):
                logging.info(
                    "%0.3f (+/-%0.03f) for %r" % (_mean, _std * 2, _params)
                )
            logging.info(f"Best score ({self.scoring}): {self.best_score_}")
        else:
            _check_hyperparameter_search_params(
                hyperparameter_search=self.hyperparameter_search
            )
        return self

    def _check_vars(self) -> None:
        _check_estimator_class(estimator_class=self.estimator_class)
        _check_scoring_function(
            scoring=self.scoring, estimator_class=self.estimator_class
        )
        _check_hyperparameter_search_params(
            hyperparameter_search=self.hyperparameter_search
        )


def hyperparameter_search_helper(
    X: typing.Union[numpy.ndarray, pandas.DataFrame],
    y: typing.Union[numpy.ndarray, pandas.Series],
    feature_names: list[str],
    estimator_class: typing.Union[
        lightgbm.LGBMClassifier,
        lightgbm.LGBMRegressor,
        sklearn.ensemble.RandomForestClassifier,
        sklearn.ensemble.RandomForestRegressor,
    ],
    estimator_params: typing.Union[None, dict[str, typing.Any]],
    scoring: str,
    n_splits: int,
    n_repeats: int,
    hyperparameter_search: typing.Union[
        None, GridSearchCV, RandomizedSearchCV
    ] = None,
    n_jobs: typing.Union[None, int] = -1,
    random_seed: int = utils.DEFAULT_RANDOM_SEED,
) -> tuple[float, dict[str, typing.Any]]:
    """
    A helper method that does hyperparameter tuning and cross-validation using a
    set of selected features. The method returns the best CV performace estimate
    along with the the hyperparameter configuration that actually achieved the
    best CV score using the set of selected features.

    Args:
        X: An input dataset assumed to contain features (columns) and \
            corresponding observations (rows).
        y: The set of samples associated with the target variable \
            of the dataset.
        feature_names: A list of selected features.
        estimator_class: The scikit-learn or Microsoft LightGBM \
            tree-based estimator to use. The estimator can either be a \
            classifier or a regressor.
        estimator_params: The estimator hyperparameters and corresponding \
            values to search or directly use. If only a single value for \
            each hyperparameter is provided then only cross-validation \
            will be performed and no hyperparameter search will be \
            performed. Defaults to None.
        scoring: The specification of a scoring function to use for \
            model-evaluation, i.e., a function that can be used for \
            assessing the prediction error of a trained model given a test \
            set.
        n_splits: The number of folds a dataset should be divided into.
        n_repeats: The number of times the division of a dataset into a \
            certain number of folds should be repeated.
        hyperparameter_search: The type of hyperparameter search method to \
            apply. Defaults to None, which simply results in the \
            cross-validation.
        n_jobs: The number of jobs to run in parallel. None means 1 while \
            -1 means use all processors. Defaults to -1.
        random_seed: The random seed to use for \
            reproducibility purposes. Defaults to \
                :const:`shapfire.utils.DEFAULT_RANDOM_SEED`.
    """
    # TODO: Check 'X' and 'y' have the same dimensions
    _X, _y = X[feature_names], y
    kfold = get_kfold_cross_validator(
        estimator_class=estimator_class,
        n_splits=n_splits,
        n_repeats=n_repeats,
    )

    hyperparameter_search_helper = HyperparameterSearchHelper(
        cv=kfold,
        estimator_class=estimator_class,
        estimator_params=estimator_params,
        random_seed=random_seed,
        hyperparameter_search=hyperparameter_search,
        n_jobs=n_jobs,
        scoring=scoring,
    )
    hyperparameter_search_helper.fit(X=_X, y=_y)

    best_score_ = hyperparameter_search_helper.best_score_
    best_params_ = hyperparameter_search_helper.best_params_
    return best_score_, best_params_  # type: ignore


# TODO: Refactor into class 'AutoHierarchicalAssociationClustering' in file
# shapfire.clutering.py
class FeatureSelectionHelper:
    """A ShapFire helper class for organizing data related to feature clusters
    and feature subsets."""

    def __init__(
        self,
        random_seed: int = utils.DEFAULT_RANDOM_SEED,
    ) -> None:
        """
        Initialize a FeatureSelectionHelper object.

        Args:
            random_seed: The random seed to use for \
                reproducibility purposes. Defaults to \
                :const:`shapfire.utils.DEFAULT_RANDOM_SEED`.
        """
        # Internal variables for easy access to data
        self._cluster_labels_df: typing.Union[None, pandas.DataFrame] = None

    @property
    def nclusters(self) -> int:
        if self._cluster_labels_df is not None:
            return numpy.unique(
                self._cluster_labels_df["cluster_label"].values
            ).shape[0]
        else:
            # TODO: No clustering has been executed
            raise ValueError("TODO")

    @property
    def largest_cluster(self) -> int:
        cluster_size_max = 0
        if self._cluster_labels_df is not None:
            for _, df in self._cluster_labels_df.groupby("cluster_label"):
                cluster_size = df.shape[0]
                if cluster_size > cluster_size_max:
                    cluster_size_max = cluster_size
            return cluster_size_max
        else:
            raise ValueError("TODO")

    @property
    def feature_clusters(self) -> list[str]:
        if self._cluster_labels_df is not None:
            feature_clusters: list[str] = []
            for _, df in self._cluster_labels_df.groupby("cluster_label"):
                feature_clusters.append(df["feature_name"].to_list())
            return feature_clusters
        else:
            raise ValueError("TODO")

    def _identify_clusters(
        self,
        X: typing.Union[numpy.ndarray, pandas.DataFrame],
    ) -> tuple[pandas.DataFrame, AutoHierarchicalAssociationClustering]:
        """
        Given a dataset containing features (columns) and corresponding \
        observations (rows) identify highly associated/correlated features by \
        grouping these into clusters.

        Args:
            X: An input dataset containing features (columns) and \
                corresponding observations (rows).

        Raises:
            ValueError: If the given input argument 'X' is not type \
                'ndarray' or 'DataFrame'.

        Returns:
            Data pertaining to the best clustering of features.
        """
        if isinstance(X, numpy.ndarray):
            logging.info("Converting input 'ndarray' 'X' to a 'DataFrame'.")
            _X = pandas.DataFrame(X)
        elif isinstance(X, pandas.DataFrame):
            _X = X.copy()
        else:
            raise TypeError(
                "The given input argument 'X' is not of type "
                + "'ndarray' or 'DataFrame'. 'X' is instead "
                + f"of type {type(X)}."
            )
        # TODO: Drop a feature (dataframe column) if more than 1/3
        #       percent of the values in the column are missing
        # TODO: Replace NAN values in a column with the mean of
        #       of the values of the feature (dataframe column)
        _X = _X.dropna(axis=0)
        (cluster_labels_df, clustering_model,) = _identify_colinear_features(
            df=_X,
        )
        self._cluster_labels_df = cluster_labels_df
        return (
            cluster_labels_df,
            clustering_model,
        )


class RankedDifferences:
    def __init__(
        self,
        reference: str = "mean",
        ascending: bool = True,
    ) -> None:
        """
        _summary_

        Args:
            reference: The data fusion method to use for producing a reference
                vector. Defaults to "mean".
            ascending: The order in which values are ranked. Defaults to True.
        """
        self.reference = reference
        self.ascending = ascending

        # Check that the given input is valid
        self._check_vars()

    def fit(self, X: pandas.DataFrame, y: None = None) -> pandas.DataFrame:
        return self.ranked_differences(
            df=X,
            ascending=self.ascending,
        )

    def ranked_differences(
        self,
        df: pandas.DataFrame,
        ascending: bool = True,
        ties: str = "average",
    ) -> pandas.DataFrame:
        # Produce a reference vector using a data fusion method over the rows
        # of a panads dataframe
        ref_vector = self.calculcate_reference(df=df)
        # Rank the data in the reference vector (a pandas series)
        ref_vector_ranked = ref_vector.rank(
            method=ties, ascending=ascending, axis=0
        )

        # Rank the data in each row of the input dataframe over the columns
        df_ranked = df.rank(method=ties, ascending=ascending, axis=1)

        # Calculate the row-wise difference between the ranked rows in the input
        # dataframe 'df' and the generated (ranked) reference vector
        diffs = df_ranked.subtract(ref_vector_ranked, axis=1)

        # Return the mean absolute distance to the (ranked) reference vector
        return diffs.abs().mean()

    def calculcate_reference(self, df: pandas.DataFrame) -> pandas.Series:
        """Produce a reference vector with a data fusion method over the
        rows."""
        if self.reference == "min":
            reference_vector = df.min(axis=0)
        elif self.reference == "max":
            reference_vector = df.max(axis=0)
        elif self.reference == "mean":
            reference_vector = df.mean(axis=0)
        elif self.reference == "median":
            reference_vector = df.median(axis=0)
        return reference_vector  # type: ignore

    def _check_vars(self) -> None:
        self.reference = self.reference.lower().strip()
        _check_reference_vector_params(reference=self.reference)


class ShapFire(BaseEstimator, TransformerMixin):
    _HISTORY_REQUIRED_FIELDS = [
        "score",
        "feature_importances",
        # Data pertaining to the following fields are not needed anywhere but
        # returned for the sake of convenience in case a user needs the data...
        "shap_values",
    ]

    def __init__(
        self,
        estimator_class: typing.Union[
            lightgbm.LGBMClassifier,
            lightgbm.LGBMRegressor,
            sklearn.ensemble.RandomForestClassifier,
            sklearn.ensemble.RandomForestRegressor,
        ],
        scoring: str,
        estimator_params: typing.Union[None, dict[str, typing.Any]] = None,
        n_splits: int = DEFAULT_SPLITS,
        n_repeats: int = DEFAULT_REPEATS,
        random_seed: int = utils.DEFAULT_RANDOM_SEED,
        iterations: typing.Union[None, int] = None,
        reference: str = "mean",
        n_samples: int = 1000,
        n_batches: int = 250,
    ) -> None:
        """
        The main class used for applying SHAP feature importance rank ensembling
        for feature selection.

        Args:
            estimator_class: The scikit-learn or Microsoft LightGBM \
                tree-based estimator to use. The estimator can either be a \
                classifier or a regressor.
            scoring: The specification of a scoring function to use for \
                model-evaluation, i.e., a function that can be used for \
                assessing the prediction error of a trained model given a test \
                set.
            estimator_params: The estimator hyperparameters and corresponding \
                values to search or directly use. If only a single value for \
                each hyperparameter is provided then only cross-validation \
                will be performed and no hyperparameter search will be \
                performed. Defaults to None.
            n_splits: The number of folds to generate in the outer loop \
                of a nested cross-validation. Defaults to \
                    :const:`shapfire.shapfire.DEFAULT_SPLITS`.
            n_repeats: The number of new folds that should be generated \
                in the outer loop of a nested cross-validation. Defaults to \
                    :const:`shapfire.shapfire.DEFAULT_REPEATS`.
            random_seed: The random seed to use for reproducibility purposes. \
                Defaults to :const:`shapfire.utils.DEFAULT_RANDOM_SEED`.
            iterations: The number of feature subsets to sample and subsequently
                use for model-training such that SHAP feature importance values
                can be extracted. Defaults to None which in turns sets the
                number of iterations to the size of the largest cluster of
                highly associated features found.
            reference: The data fusion method to use for producing a reference
                vector. Defaults to "mean".
            n_samples: The number of random samples of rank permutations to use
                in a batch. Several batches of random samples are used to
                estimate a "ranking distribution" which in turn is used to
                determine a feature importance cut-off threshold. Defaults to
                1000.
            n_batches: The number of batches of random samples that should be
                used to estimate a "ranking distribution" which in turn is used
                to determine a feature importance cut-off threshold. Defaults to
                250.

        Attributes:
            ranked_differences: A class attribute and pandas dataframe that
                specifies the final importance values associated with each
                of the features in the given input dataset.
            selected_features: A ShapFire class attribute and list that
                specifies the final feature subset selected by ShapFire and
                which is expected to achieve the best possible model
                performance.
        """
        # Class vars corresponding to input args
        self.estimator_class = estimator_class
        self.scoring = scoring
        self.estimator_params = estimator_params
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.iterations = iterations
        self.random_seed = random_seed
        self.reference = reference
        self.n_samples = n_samples
        self.n_batches = n_batches

        # Check that the given input is valid
        self._check_vars()

        # Set random seed for reproducibility purposes
        numpy.random.seed(self.random_seed)

        # Public accessible vars associated with the most important features
        # These vars wil eventually be set after a call to 'fit()'
        self.ranked_differences: typing.Union[None, pandas.DataFrame] = None
        self.selected_features: typing.Union[None, list[str]] = None

        # Internal vars for easy access to data associated with the importance
        # ranking of features
        self._history: pandas.DataFrame = pandas.DataFrame()
        self._feature_selector: typing.Union[
            None, FeatureSelectionHelper
        ] = None
        # Private class variable for a progress bar that is to be updated
        # TODO: Determine progrss bar type
        self._progress_bar: typing.Union[None, typing.Any] = None

        # Keep a class variable around to store a plotting interface object
        # such that all necessary plotting methods can be accessed through it
        self._plotting_interface: typing.Union[
            None, ShapFirePlottingInterface
        ] = None

    def fit(
        self,
        X: typing.Union[numpy.ndarray, pandas.DataFrame],
        y: typing.Union[numpy.ndarray, pandas.DataFrame],
    ) -> "ShapFire":
        """
        Perform SHAP feature importance rank ensembling for the purpose of
        ranking and selecting the features that can be said to be the most
        important for a certain prediction task at hand.

        Args:
            X: An input dataset that the ShapFire method should be applied to. \
                The dataset is assumed to contain features (columns) and \
                corresponding observations (rows).
            y: The samples associated with the target variable of the dataset.

        Returns:
            A ShapFire object containing the necessary data associated with \
                the most important features of the given input dataset.
        """
        if isinstance(X, numpy.ndarray):
            logging.info("Converting input 'ndarray' 'X' to a 'DataFrame'.")
            columns = [f"{i}" for i in range(X.shape[1])]
            X = pandas.DataFrame(data=X, columns=columns)
        elif isinstance(X, pandas.DataFrame):
            # Make sure the column names are strings!
            X.columns = [str(name) for name in X.columns]
        else:
            raise TypeError(
                "The given input argument 'X' is not of type "
                + "'ndarray' or 'DataFrame'. 'X' is instead "
                + f"of type {type(X)}."
            )
        if isinstance(y, numpy.ndarray):
            logging.info("Converting input 'ndarray' 'y' to a 'Series'.")
            y = pandas.Series(y)
        elif isinstance(y, pandas.Series):
            pass
        else:
            raise TypeError(
                "The given input argument 'X' is not of type "
                + "'ndarray' or 'Series'. 'X' is instead "
                + f"of type {type(X)}."
            )

        # Determine feature clustering
        self._feature_selector = FeatureSelectionHelper(
            random_seed=self.random_seed,
        )
        (
            cluster_labels_df,
            clustering_model,
        ) = self._feature_selector._identify_clusters(X=X)

        if self.iterations is None:
            self.iterations = self._feature_selector.largest_cluster

        # Create progress bar which will be updated continuously to track the
        # progress of the outer cross-validation loop
        total = self.n_repeats * self.n_splits * self.iterations
        self._progress_bar = tqdm(
            total=total,
            unit_scale=True,
            ascii=" >=",
            bar_format="{desc:<20}{percentage:3.0f}%|{bar:25}{r_bar}",
            desc="ShapFire progress",
        )

        # Perform repeated nested Cross-Validation (CV):
        self._outer_cv_loop(X=X, y=y)

        # Close/stop the progress bar
        self._progress_bar.close()

        # Calculate normalized SHAP feature importance scores and pick the best
        # feature from each of the previously found clusters
        df = self._calculate_normalized_shap_feature_importance()

        # Extract and organize data associated with each tested feature subset
        data_dict = self._reorganize_feature_importance_values(df=df)

        # TODO: Adjust feature selection strategy!
        # ndf = self._calculate_ranked_differences(data_dict=data_dict)
        # self._discard_unimportant_feautres(df=ndf)
        feature_ranking_df = self._pick_top_k_from_clusters(
            df=df,
            top_k=None,
        )

        # TODO: 
        # Select all features from each cluster
        selected_features = feature_ranking_df.groupby(by=["cluster"]).head(1)

        # Discard features with 0 importance. A selected feature should
        # not have 0 importance!
        selected_features = selected_features[
            selected_features["normalized_feature_importance"] > 0
        ]

        # Further filtering based on a cutoff value
        cut_value = self._find_cutoff(df=selected_features)
        print("SHAP importance cutoff value: ", cut_value)
        selected_features = selected_features[
            selected_features["normalized_feature_importance"] >=  cut_value
        ]

        selected_feature_names = [v[0] for v in selected_features.index.values]
        self.selected_features = selected_feature_names
        return self

    def transform(
        self, X: typing.Union[numpy.ndarray, pandas.DataFrame]
    ) -> typing.Union[numpy.ndarray, pandas.DataFrame]:
        """
        Reduce the input dataset X containing features (columns) and
        corresponding observations (rows), to only the columns of features
        selected by ShapFire.

        Args:
            X: The original input dataset that the ShapFire method was applied \
                to. The dataset is assumed to contain features (columns) and \
                corresponding observations (rows).

        Raises:
            ValueError: If the :meth:`fit` method has not yet been called.

        Returns:
            A reduced dataset that only contains the most important features \
                (columns).
        """
        if self.selected_features is not None:
            return X[self.selected_features]
        else:
            raise ValueError(
                "Method '.fit(X, y)' has not yet been called. "
                + "Simply call method '.fit_transform(X, y)' or call "
                + "'.fit(X, y)' before calling '.transform(X)'."
            )

    def fit_transform(
        self,
        X: typing.Union[numpy.ndarray, pandas.DataFrame],
        y: typing.Union[numpy.ndarray, pandas.DataFrame],
    ) -> typing.Union[numpy.ndarray, pandas.DataFrame]:
        """
        Perform SHAP feature importance rank ensembling for the purpose of
        selecting the features that are the most important. Subsequently, reduce
        the input dataset 'X' to only the columns of the selected features.

        Args:
            X: An input dataset that the ShapFire method should be applied to. \
                The dataset is assumed to contain features (columns) and \
                corresponding observations (rows).
            y: The samples associated with the target variable of the dataset.

        Returns:
            A reduced dataset that only contains the data associated with the \
                most important features.
        """
        self.fit(X=X, y=y)
        return self.transform(X=X)

    def plot_ranking(
        self,
        groupby: str = "cluster",
        rcParams: typing.Union[None, dict[str, str]] = None,
        figsize: typing.Union[None, tuple[float, float]] = None,
        fontsize: int = 10,
        with_text: bool = True,
        with_overlay: bool = True,
        ax: typing.Union[None, Axes] = None,
    ) -> tuple[Figure, Axes]:
        """
        Plot the feature importance scores associated with each feature. The
        features will be ordered in the figure from best to worst and possibly
        according to which cluster they each belong to.

        Args:
            groupby: A string value indicating how the feature importance \
                ranking should be displayed in a figure. If the option \
                'cluster' is chosen, then the features are grouped and \
                shown in the figure based on their assigned cluster and \
                according to the importance rank of the best feautre in the \
                cluster. If 'feature' is chosen, then the features are \
                shown in the figure purely according to their global rank \
                without any consideration to what cluster each features are a \
                part of.
            figsize: The width and height of the figure in inches. Defaults to \
                None.
            fontsize: The size of the font present in the figure. Defaults to \
                10.
            with_text: If input argument :code:`groupby` is set to \
                'cluster', then :code:`with_text` determines whether \
                features that have been grouped in the figure by the cluster \
                they each belong to, should also be annotated with a text \
                label. Defaults to True.
            with_overlay: Depending on whether :code:`groupby` is set to \
                'cluster' or 'feature', groups of features or individual \
                features are assigned a gray-scale overlay creating a visual \
                grouping / delimitation of features. Defaults to True.
            ax: A Matplotlib Axes object. Defaults to None.

        Returns:
            A Matplotlib Figure and Axes object.
        """
        if self._plotting_interface is None:
            self._plotting_interface = ShapFirePlottingInterface(shapfire=self)
            # TODO: If fit is called again, then self._plotting_interface should
            #       be set to None.
        return self._plotting_interface.plot_ranking(
            groupby=groupby,
            rcParams=rcParams,
            figsize=figsize,
            fontsize=fontsize,
            with_text=with_text,
            with_overlay=with_overlay,
            ax=ax,
        )

    def plot_importance(
        self,
        plot_type: str = "stripplot",
        groupby: str = "cluster",
        rcParams: typing.Union[None, dict[str, str]] = None,
        figsize: typing.Union[None, tuple[float, float]] = None,
        fontsize: int = 10,
        with_text: bool = True,
        with_overlay: bool = True,
        ax: typing.Union[None, Axes] = None,
    ) -> tuple[Figure, Axes]:
        """
        Plot the normalized SHAP feature importance scores associated with each
        feature. The features will be ordered in the figure from best to worst
        and possibly according to which cluster they each belong to.

        Args:
            plot_type: An argument to control how the normalized SHAP feature \
                importance scores should be displayed. Defaults to 'stripplot'.
            groupby: A string value indicating how the feature importance \
                ranking should be displayed in a figure. If the option \
                'cluster' is chosen, then the features are grouped and \
                shown in the figure based on their assigned cluster and \
                according to the importance rank of the best feautre in the \
                cluster. If 'feature' is chosen, then the features are \
                shown in the figure purely according to their global rank \
                without any consideration to what cluster each features are a \
                part of.
            figsize: The width and height of the figure in inches. Defaults to \
                None.
            fontsize: The size of the font present in the figure. Defaults to \
                10.
            with_text: If input argument :code:`groupby` is set to \
                'cluster', then :code:`with_text` determines whether \
                features that have been grouped in the figure by the cluster \
                they each belong to, should also be annotated with a text \
                label. Defaults to True.
            with_overlay: Depending on whether :code:`groupby` is set to \
                'cluster' or 'feature', groups of features or individual \
                features are assigned a gray-scale overlay creating a visual \
                grouping / delimitation of features. Defaults to True.
            ax: A Matplotlib Axes object. Defaults to None.

        Returns:
            A Matplotlib Figure and Axes object.
        """
        if self._plotting_interface is None:
            self._plotting_interface = ShapFirePlottingInterface(shapfire=self)
            # TODO: If fit is called again, then self._plotting_interface should
            #       be set to None.
        return self._plotting_interface.plot_importance(
            plot_type=plot_type,
            groupby=groupby,
            rcParams=rcParams,
            figsize=figsize,
            fontsize=fontsize,
            with_text=with_text,
            with_overlay=with_overlay,
            ax=ax,
        )

    def _check_vars(self) -> None:
        _check_scoring_function(
            scoring=self.scoring, estimator_class=self.estimator_class
        )
        if isinstance(self.scoring, str):
            self.scoring = self.scoring.strip().lower()
        _check_cv_params(n_splits=self.n_splits, n_repeats=self.n_repeats)
        if self.iterations is not None:
            if self.iterations < 1:
                raise ValueError(
                    "The given input argument 'iterations' can not be"
                    "less than 1."
                )

    # TODO: Make public and not part of class
    def _get_score(
        self,
        estimator: typing.Union[
            lightgbm.LGBMClassifier,
            lightgbm.LGBMRegressor,
            sklearn.ensemble.RandomForestClassifier,
            sklearn.ensemble.RandomForestRegressor,
        ],
        X_test: pandas.DataFrame,
        y_test: pandas.Series,
    ) -> typing.Union[None, dict[str, typing.Any]]:
        """
        Retrieve the performance score of an estimator on a given test set.

        Args:
            estimator: A LightGBM estimator from Microsoft's LightGBM \
                gradient boosting decision tree framework. The estimator can \
                either be a classifier or a regressor. The estimator is \
                assumed to have been trained on a training dataset and \
                should be evaluated on a test dataset.
            X_test: A test dataset.
            y_test: The samples associated with the target variable of the \
                test dataset.

        Raises:
            ValueError: If the estimator can not be identified as being a \
                classifier or regressor.

        Returns:
            Returns a dictionary with a performance score and possibly \
            additional data pertaining to a certain type of performance score.
        """
        dict_ = {}
        if is_classifier(self.estimator_class):
            # Handle special scoring functions where additional data, beyond
            # just a score,  needs to be saved and passed on
            if self.scoring == "roc_auc":
                fpr, tpr, roc_auc = get_roc_auc_statistics(
                    estimator=estimator,
                    X_test=X_test,
                    y_test=y_test,
                )
                dict_["fpr"] = fpr
                dict_["tpr"] = tpr
                dict_["roc_auc"] = roc_auc
                return dict_
            else:
                raise ValueError("TODO: Not yet implemented!")
        elif is_regressor(self.estimator_class):
            raise ValueError("TODO: Not yet implemented!")
        else:
            raise ValueError(
                "It could not be determined whether the given "
                + f"'estimator': {estimator} is a classifier or a regressor."
            )

    def _outer_cv_loop(
        self,
        X: pandas.DataFrame,
        y: pandas.Series,
    ) -> None:
        """
        Given a dataset perform repeated cross-validation to estimate SHAP
        values and thus the importance of the different features that are
        contained in the input dataset.

        Args:
            X: The original input dataset that the ShapFire method is applied \
                to. The dataset is assumed to contain features (columns) and \
                corresponding observations (rows).
            y: The original set of samples associated with the target variable \
                of the dataset.
            cv: A scikit-learn cross-validator class for generating train/test \
                folds.

        Raises:
            NotImplementedError: If a not yet implemented scoring function is \
                passed as an argument.
        """
        history = []
        repeat_number = 1

        cv = get_kfold_cross_validator(
            estimator_class=self.estimator_class,
            n_repeats=self.n_repeats,
            n_splits=self.n_splits,
        )
        feature_clusters = list(
            self._feature_selector.feature_clusters  # type: ignore
        )
        cs = ClusterSampler(feature_clusters=feature_clusters)

        for _ in range(self.iterations):  # type: ignore
            selected_features = cs.sample_feature_subset()
            for i, (train_ix, test_ix) in enumerate(cv.split(X=X, y=y)):
                X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
                y_train, y_test = y.values[train_ix], y.values[test_ix]

                _X_train, _y_train = X_train[selected_features], y_train
                estimator = self.estimator_class(
                    random_state=self.random_seed,
                ).fit(
                    X=_X_train,
                    y=_y_train.ravel(),
                )

                # Retrieve SHAP values on outer loop CV test set using
                # best estimator refitted on inner loop CV training + test set
                shap_values = shap.TreeExplainer(estimator).shap_values(
                    X_test[selected_features]
                )
                # TODO: Make sure shap_values[1] can actually be accessed!
                #       else raise error!
                values = numpy.abs(shap_values[1]).mean(axis=0)
                feature_importances = pandas.DataFrame(
                    list(zip(selected_features, values)),
                    columns=["feature_name", "feature_importance"],
                )
                feature_importances.sort_values(
                    by=["feature_importance"],
                    ascending=False,
                    inplace=True,
                )
                score: typing.Union[
                    None, dict[str, typing.Any]
                ] = self._get_score(
                    estimator=estimator,
                    X_test=X_test[selected_features],
                    y_test=y_test,
                )

                if score is None:
                    raise NotImplementedError(
                        f"The scorer '{self.scoring}' has not yet been "
                        + "implemented for use with ShapFire."
                    )
                dict_ = {
                    # 'score' a dictionary that contains data pertaining to
                    # the estimate of the performance on the outer loop CV test
                    # set using a certain scoring measure specified by
                    # 'self.scoring'.
                    "score": score,
                    # 'feature_importance' is dataframe that contains feature
                    # names and corresponding importance values for each feature
                    # selected in the inner CV loop.
                    "feature_importances": feature_importances,
                    # 'shap_values' contains the raw numpy array output from the
                    # SHAP Python library.
                    "shap_values": shap_values,
                    "repeat_number": repeat_number,
                }
                history.append(dict_)
                if ((i + 1) % self.n_splits) == 0:
                    repeat_number += 1

                # Update the progress bar
                self._progress_bar.update(1)  # type: ignore

        _history = pandas.DataFrame(data=history)
        # If the current ShapFire object already has a 'self._history'
        # defined then reset the dataframe so data does not accumulate
        if self._history is not None:
            self._history = pandas.DataFrame()
        self._history = pandas.concat(
            [
                self._history.reset_index(drop=True),
                _history.reset_index(drop=True),
            ],
            ignore_index=True,
            join="outer",
            axis=0,
        )

    def _calculate_normalized_shap_feature_importance(self) -> pandas.DataFrame:
        """
        Calculate and organize the normalized SHAP feature importance each test
        fold in the cross-validation loop. Normalizing SHAP feature importance
        scores makes it possible to compare and aggregate results across
        different folds if necessary.

        Raises:
            ValueError: If the internal class variable  '._history' is None.
            ValueError: If the internal class variable '._feature_selector' \
                is None.
            ValueError: If the internal class variable \
                '._feature_selector._df_cluster_labels' is None.
            ValueError: If a certain column name is not contained in the \
                internally used '._history' pandas dataframe.

        Returns:
            A pandas dataframe that contains normalized SHAP feature importance
            values associated with each feature in a tested feature subset.
        """
        # Validate and check necessary data before proceeding
        if self._history is None:
            raise ValueError(
                "Internal error. The internal class variable "
                + "'._history' is None. This should not happend if "
                + "the method is called via the '.fit(X, y)' method."
            )
        if self._feature_selector is None:
            raise ValueError(
                "Internal error. The internal class variable "
                + "'._feature_selector' is None. This should not happend if "
                + "the method is called via the '.fit(X, y)' method."
            )
        if self._feature_selector._cluster_labels_df is None:
            raise ValueError(
                "Internal error. The internal class variable "
                + "'._feature_selector._df_cluster_labels ' is None. This "
                + "should not happend if the method is called via the "
                + "'.fit(X, y)' method."
            )

        # Verify that all required data is contained in 'self._history'
        for column_name in self._HISTORY_REQUIRED_FIELDS:
            if column_name not in self._history.columns:
                raise ValueError(
                    f"The column name {column_name} is required but is "
                    + "not contained in the internally used "
                    + "'._history' pandas dataframe."
                )
        folds: int = self._history.shape[0]
        arr = []
        for i in range(folds):
            df_fold = (
                self._history["feature_importances"]
                .iloc[i]
                .reset_index(drop=True)
            )
            score: float = self._history["score"].iloc[i][self.scoring]

            # Sum feature importance value such that we can compute a
            # normalized feature importance value that lies in the range
            # [0, 1]. This makes it possible to then aggregate and compare
            # scores across differrent trained models.
            total = df_fold["feature_importance"].sum()

            # Create new column with normalized feature importance scores
            df_fold["normalized_feature_importance"] = (
                df_fold["feature_importance"] / total
            )

            # Enumerate CV folds from 1...
            df_fold.index = df_fold.index + 1
            for index, row in df_fold.iterrows():
                d = {
                    "test_fold": i + 1,
                    # Set the feature name
                    "feature_name": row["feature_name"],
                    # Set the normalized feature importance score calculated
                    # based on the outer loop CV test fold
                    "normalized_feature_importance": row[
                        "normalized_feature_importance"
                    ],
                    # Set the rank of the feature. The rank is based on the
                    # computed 'normalized_feature_importance'
                    "feature_rank": index,
                    # Set the performance score that was calculated based on
                    # the outer loop CV test fold
                    "score": score,
                    # Retrieve the cluster that the feature belongs to
                    "cluster": self._feature_selector._cluster_labels_df[
                        self._feature_selector._cluster_labels_df[
                            "feature_name"
                        ]
                        == row["feature_name"]
                    ]["cluster_label"].iat[0],
                }
                arr.append(d)
        return pandas.DataFrame(data=arr)



    def _pick_top_k_from_clusters(
        self,
        df: pandas.DataFrame,
        top_k: typing.Union[None, int] = None,
    ) -> pandas.DataFrame:
        """
        The method picks the top k best features, ranked by their normalized
        SHAP importance score, from each cluster of highly associated/correlated
        features.

        Args:
            df: A dataframe containing normalized SHAP feature importance \
                scores that can be used for ranking the importance of the \
                different features.
            top_k: The numer of features to pick from each cluster of \
                features. Defaults to None.

        Raises:
            TypeError: If the 'df' input argument is not a pandas dataframe.
            ValueError: If the 'top_k' input argument is not an integer value.

        Returns:
            A dataframe reduced to the top k features from each cluster of \
            highly associated/correlated features, ranked by their associated \
            normalized SHAP feature importance.
        """
        REQUIRED_FIELDS = [
            "feature_name",
            "cluster",
            "normalized_feature_importance",
        ]
        # Validate input arguments before proceeding
        if not isinstance(df, pandas.DataFrame):
            raise TypeError(
                "The internally passed input argument 'df' is not of type "
                + f"'DataFrame'. 'df' is instead of type {type(df)}."
            )
        else:
            # Verify that all required data is contained in input argument 'df'
            for column_name in REQUIRED_FIELDS:
                if column_name not in list(df.columns):
                    raise ValueError(
                        f"The column name {column_name} is required but is "
                        + "not contained in the internally passed input "
                        + "argument 'df' pandas dataframe."
                    )

        # Extract necessary data
        # TODO: Maybe make it possible to choose between agg("median") and
        #       agg("mean")?
        _df = (
            df[REQUIRED_FIELDS]
            .groupby(by=["feature_name", "cluster"])
            .agg("median")
            .sort_values(
                by=["normalized_feature_importance"],
                ascending=False,
            )
            .groupby(by=["cluster"])
        )
        # if top_k is not None:
        #     # Return the top k best ranked features from each cluster
        #     return _df.head(top_k)
        # else:
        #     # Return all features from each cluster
        #     return _df.head(numpy.inf)

        if top_k is not None:
            # Return the top k best ranked features from each cluster
            return _df.head(top_k)
        else:
            # Return all features from each cluster
            return _df.head(numpy.inf)

    def _find_cutoff(self, df, relative_change=0.1):
        ndf = df.groupby(
            level=0
        ).apply(
            max
        ).sort_values(
            by="normalized_feature_importance",
            ascending=False,
        )

        values = ndf.values.flatten()
        features = ndf.index.values.flatten()
        ys = []
        for i in range(1, len(values) + 1):
            sum1 = values[:i]
            sum2 = values[i:]
            norm_sum1 = numpy.sum(sum1)
            norm_sum2 = numpy.sum(sum2)
            # Interpretation is: how much more does remaining feature contributions 
            # "norm_sum2" explain compared to current total contributions "norm_sum1"
            # calculated from the i - n first features
            proportion = ((norm_sum2 / norm_sum1) * 100)
            ys.append(proportion)
        final_df = pandas.DataFrame(data = ys, columns=["proportions"])
        final_df.index = features
        print("Final df: ")
        print(final_df)
        cut_feature = final_df[final_df["proportions"] >= relative_change].index[-1]
        temp_df = df.droplevel(1)
        cut_value = temp_df[temp_df.index == cut_feature]["normalized_feature_importance"].iloc[0]
        return cut_value

    def _reorganize_feature_importance_values(
        self, df: pandas.DataFrame
    ) -> dict[str, pandas.DataFrame]:
        # Organize data per tested feature subset
        fsc = FeatureSubsetCollection()
        for _, _df in df.groupby("test_fold"):
            reduced_df = _df[
                [
                    "test_fold",
                    "feature_name",
                    "normalized_feature_importance",
                ]
            ]
            pivot_df = reduced_df.pivot(
                index=["test_fold"],
                columns=["feature_name"],
                values=["normalized_feature_importance"],
            )
            pivot_df = pivot_df["normalized_feature_importance"].reset_index(
                drop=True
            )
            pivot_df.columns.name = None
            names = list(pivot_df.columns)
            feature_names = sorted(names)
            key = "-".join(feature_names)
            fsc._add_entries(key, pivot_df)
        return fsc._data_dict

    def _calculate_ranked_differences(
        self, data_dict: dict[str, pandas.DataFrame]
    ) -> pandas.DataFrame:
        # For each evaluated subset of features calculate the ranked differences
        # between rankings obtained from SHAP values associated with the feature
        # subsets and reference vectors produced based on the same data through
        # a data fusion method
        evaluated_feature_subsets = []
        for key in data_dict:
            feature_importance_values = data_dict[key]
            if feature_importance_values is not None:
                ranked_differences = RankedDifferences(
                    reference=self.reference, ascending=False
                ).fit(feature_importance_values)
                ranked_differences = ranked_differences.to_dict()
                ranked_differences[
                    "nsamples"
                ] = feature_importance_values.shape[0]
                evaluated_feature_subsets.append(ranked_differences)
        df = pandas.DataFrame(data=evaluated_feature_subsets)
        nsamples = df["nsamples"]
        ndf = df.drop("nsamples", axis=1)
        # Calculate weighted averages
        ndf = ndf.multiply(nsamples, axis="rows").sum() / numpy.sum(nsamples)
        ndf = ndf.sort_values(ascending=True).to_frame("ranked_difference")
        return ndf

    def _discard_unimportant_feautres(self, df: pandas.DataFrame) -> None:
        # Determine a feature importance cut-off threshold
        self.threshold_finder = utils.ThresholdFinder(
            random_seed=self.random_seed,
            ncols=self._feature_selector.nclusters,  # type: ignore
            n_batches=self.n_batches,
            n_samples=self.n_samples,
        )
        self.threshold_finder.fit()

        cluster_labels = []
        for index, _ in df.iterrows():
            cluster_label = self._loopkup_cluster_label(feature_name=index)
            cluster_labels.append(cluster_label)
        df["cluster_label"] = cluster_labels
        self.ranked_differences = df

        selected_features = df[
            df["ranked_difference"] <= self.threshold_finder.lower_threshold
        ]
        print("THIS ONE IS RUN!")
        self.selected_features = selected_features.index.to_list()

    def _loopkup_cluster_label(self, feature_name: str) -> str:
        # Retrieve the cluster that the given input feature 'feature_name'
        # belongs to
        df = self._feature_selector._cluster_labels_df  # type: ignore
        return df[  # type: ignore
            df["feature_name"] == feature_name  # type: ignore
        ]["cluster_label"].iat[0]


class FeatureSubsetCollection:
    def __init__(self) -> None:
        self.feature_subsets: list = []
        self._data_dict: dict[str, pandas.DataFrame] = {}

    def _add_entries(self, key: str, data: pandas.DataFrame) -> None:
        if key in self._data_dict:
            self._data_dict[key] = pandas.concat(
                [
                    self._data_dict[key].reset_index(drop=True),
                    data.reset_index(drop=True),
                ],
                ignore_index=True,
                join="outer",
                axis=0,
            )
        else:
            self._data_dict[key] = data


class RefitHelper:
    def __init__(
        self,
        feature_names: list[str],
        estimator_class: typing.Union[
            lightgbm.LGBMClassifier,
            lightgbm.LGBMRegressor,
            sklearn.ensemble.RandomForestClassifier,
            sklearn.ensemble.RandomForestRegressor,
        ],
        scoring: str,
        estimator_params: typing.Union[None, dict[str, typing.Any]],
        n_splits: int = DEFAULT_SPLITS,
        n_repeats: int = DEFAULT_REPEATS,
        random_seed: int = utils.DEFAULT_RANDOM_SEED,
    ) -> None:
        """
        Args:
            feature_names: A list of selected features.
            estimator_class: The scikit-learn or Microsoft LightGBM \
                tree-based estimator to use. The estimator can either be a \
                classifier or a regressor.
            scoring: The specification of a scoring function to use for \
                model-evaluation, i.e., a function that can be used for \
                assessing the prediction error of a trained model given a test \
                set.
            estimator_params: The estimator hyperparameters and corresponding \
                values to search or directly use. If only a single value for \
                each hyperparameter is provided then only cross-validation \
                will be performed and no hyperparameter search will be \
                performed. Defaults to None.
            n_splits: The number of folds to generate in the outer loop \
                of a nested cross-validation. Defaults to \
                    :const:`shapfire.shapfire.DEFAULT_SPLITS`.
            n_repeats: The number of new folds that should be generated \
                in the outer loop of a nested cross-validation. Defaults to \
                    :const:`shapfire.shapfire.DEFAULT_REPEATS`.
            random_seed: The random seed to use for reproducibility purposes. \
                Defaults to :const:`shapfire.utils.DEFAULT_RANDOM_SEED`.

        Attributes:
            history: A class attribute and pandas dataframe that contains the
                performance score (and possibly other data) associated with each
                test fold in a repeated corss-validation.
        """
        # Class vars corresponding to input args
        self.estimator_class = estimator_class
        self.scoring = scoring
        self.estimator_params = estimator_params
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.feature_names = feature_names
        self.random_seed = random_seed

        # Check that the given input is valid
        self._check_vars()

        # Set random seed for reproducibility purposes
        numpy.random.seed(self.random_seed)

        # Public accessible vars associated with the most important features
        # These vars wil eventually be set after a call to 'fit()'
        self.history = pandas.DataFrame()

    def fit(self, X: pandas.DataFrame, y: pandas.Series) -> "RefitHelper":
        history: list[dict[str, typing.Any]] = []
        repeat_number = 1

        cv = get_kfold_cross_validator(
            estimator_class=self.estimator_class,
            n_repeats=self.n_repeats,
            n_splits=self.n_splits,
        )

        for i, (train_ix, test_ix) in enumerate(cv.split(X=X, y=y)):
            X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
            y_train, y_test = y.values[train_ix], y.values[test_ix]

            _X_train, _y_train = X_train[self.feature_names], y_train
            estimator = self.estimator_class(
                random_state=self.random_seed,
                **self.estimator_params,
            ).fit(
                X=_X_train,
                y=_y_train.ravel(),
            )

            score: typing.Union[None, dict[str, typing.Any]] = self._get_score(
                estimator=estimator,
                X_test=X_test[self.feature_names],
                y_test=y_test,
            )

            if score is None:
                raise NotImplementedError(
                    f"The scorer '{self.scoring}' has not yet been "
                    + "implemented for use with ShapFire."
                )
            dict_ = {
                # 'score' a dictionary that contains data pertaining to
                # the estimate of the performance on the outer loop CV test
                # set using a certain scoring measure specified by
                # 'self.scoring'.
                "score": score,
                "repeat_number": repeat_number,
            }
            history.append(dict_)
            if ((i + 1) % self.n_splits) == 0:
                repeat_number += 1

        _history = pandas.DataFrame(data=history)
        # If the current ShapFire object already has a 'self.history'
        # defined then reset the dataframe so data does not accumulate
        if self.history is not None:
            self.history = pandas.DataFrame()
        self.history = pandas.concat(
            [
                self.history.reset_index(drop=True),
                _history.reset_index(drop=True),
            ],
            ignore_index=True,
            join="outer",
            axis=0,
        )
        return self

    def _get_score(
        self,
        estimator: typing.Union[
            lightgbm.LGBMClassifier,
            lightgbm.LGBMRegressor,
            sklearn.ensemble.RandomForestClassifier,
            sklearn.ensemble.RandomForestRegressor,
        ],
        X_test: pandas.DataFrame,
        y_test: pandas.Series,
    ) -> typing.Union[None, dict[str, typing.Any]]:
        """
        Retrieve the performance score of an estimator on a given test set.

        Args:
            estimator: A LightGBM estimator from Microsoft's LightGBM \
                gradient boosting decision tree framework. The estimator can \
                either be a classifier or a regressor. The estimator is \
                assumed to have been trained on a training dataset and \
                should be evaluated on a test dataset.
            X_test: A test dataset.
            y_test: The samples associated with the target variable of the \
                test dataset.

        Raises:
            ValueError: If the estimator can not be identified as being a \
                classifier or regressor.

        Returns:
            Returns a dictionary with a performance score and possibly \
            additional data pertaining to a certain type of performance score.
        """
        dict_ = {}
        if is_classifier(self.estimator_class):
            # Handle special scoring functions where additional data, beyond
            # just a score, needs to be saved and passed on
            if self.scoring == "roc_auc":
                fpr, tpr, roc_auc = get_roc_auc_statistics(
                    estimator=estimator,
                    X_test=X_test,
                    y_test=y_test,
                )
                dict_["fpr"] = fpr
                dict_["tpr"] = tpr
                dict_["roc_auc"] = roc_auc
                return dict_
            else:
                raise ValueError("TODO: Not yet implemented!")
        elif is_regressor(self.estimator_class):
            raise ValueError("TODO: Not yet implemented!")
        else:
            raise ValueError(
                "It could not be determined whether the given "
                + f"'estimator': {estimator} is a classifier or a regressor."
            )

    def _check_vars(self):
        _check_estimator_class(estimator_class=self.estimator_class)
        _check_scoring_function(
            estimator_class=self.estimator_class,
            scoring=self.scoring,
        )
        _check_cv_params(n_splits=self.n_splits, n_repeats=self.n_repeats)
