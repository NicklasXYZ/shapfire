"""This module contains the implementation of the ShapFire method for
feature ranking and selection."""

import faulthandler

faulthandler.enable()

import logging
import typing

import shapfire.utils as utils
from shapfire.clustering import (
    AutoHierarchicalAssociationClustering,
    ClusterSampler,
    # _identify_colinear_features,
)

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    is_classifier,
    is_regressor,
)
# from sklearn.metrics import auc, roc_curve, recall_score, confusion_matrix
from sklearn.metrics import auc, roc_curve, confusion_matrix

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
)

import numpy
import pandas
# import shap
from lightgbm import (
    LGBMClassifier,
    LGBMRegressor,
)

# from shap import TreeExplainer
# from sklearn.ensemble import (
#     RandomForestClassifier,
#     RandomForestRegressor,
# )



# from tqdm import tqdm



# from shapfire.plotting import ShapFirePlottingInterface

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
    LGBMClassifier,
    LGBMRegressor,
    # RandomForestClassifier,
    # RandomForestRegressor,
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
    },
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
        LGBMClassifier,
        LGBMRegressor,
        # RandomForestClassifier,
        # RandomForestRegressor,
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
        LGBMClassifier,
        LGBMRegressor,
        # RandomForestClassifier,
        # RandomForestRegressor,
    ],
) -> None:
    # if 'scoring' is a string then make sure the specified
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
            "It is currently not possible to pass a callable scoring " + "function"
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
        raise ValueError("The given input argument 'n_splits' can not be less than 2.")
    if n_repeats < 1:
        raise ValueError("The given input argument 'n_repeats' can not be less than 1.")


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
        LGBMClassifier,
        LGBMRegressor,
        # RandomForestClassifier,
        # RandomForestRegressor,
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
    estimator: LGBMClassifier,
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


def get_conf_matrices(estimator, X_test, y_test, n_points=20):
    # TP = confusion[1,1] is true positives
    # TN = confusion[0,0] is true negatives
    # FP = confusion[0,1] is false positives
    # FN = confusion[1,0] is false negatives
    increment = 1 / n_points
    thresholds = [i * increment for i in range(n_points + 1)]
    results = []
    for probability in thresholds:
        y_pred = (estimator.predict_proba(X_test)[:, 1] >= probability).astype(bool)
        conf_matrix = confusion_matrix(y_test.astype(bool), y_pred)
        total = numpy.sum(numpy.sum(conf_matrix))
        accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / total
        specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
        youden_index = sensitivity + specificity - 1
        results.append(
            {
                "true_positives": conf_matrix[1, 1],
                "true_negatives": conf_matrix[0, 0],
                "false_positives": conf_matrix[0, 1],
                "false_negatives": conf_matrix[1, 0],
                "accuracy": accuracy,
                "probability": probability,
                "specitivity": specificity,
                "sensitivity": sensitivity,
                "youden_index": youden_index,
            }
        )
    return results


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
            LGBMClassifier,
            LGBMRegressor,
            # RandomForestClassifier,
            # RandomForestRegressor,
        ],
        estimator_params: typing.Union[None, dict[str, typing.Any]],
        scoring: str,
        hyperparameter_search: typing.Union[
            None, GridSearchCV, RandomizedSearchCV
        ] = None,
        n_jobs: typing.Union[None, int] = None,
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
        # self._check_vars()

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
            "n_jobs": None,  # self.n_jobs,
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
            args["estimator"] = self.estimator_class(random_state=self.random_seed)
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
            for _mean, _std, _params in zip(means, stds, [params]):  # noqa: FKA01
                logging.info("%0.3f (+/-%0.03f) for %r" % (_mean, _std * 2, _params))
            logging.info(f"Best score ({self.scoring}): {self.best_score_}")
        else:
            _check_hyperparameter_search_params(
                hyperparameter_search=self.hyperparameter_search
            )
        return self

    # def _check_vars(self) -> None:
    #     _check_estimator_class(estimator_class=self.estimator_class)
    #     _check_scoring_function(
    #         scoring=self.scoring, estimator_class=self.estimator_class
    #     )
    #     _check_hyperparameter_search_params(
    #         hyperparameter_search=self.hyperparameter_search
    #     )


def hyperparameter_search_helper(
    X: typing.Union[numpy.ndarray, pandas.DataFrame],
    y: typing.Union[numpy.ndarray, pandas.Series],
    feature_names: list[str],
    estimator_class: typing.Union[
        LGBMClassifier,
        LGBMRegressor,
        # RandomForestClassifier,
        # RandomForestRegressor,
    ],
    estimator_params: typing.Union[None, dict[str, typing.Any]],
    scoring: str,
    n_splits: int,
    n_repeats: int,
    hyperparameter_search: typing.Union[None, GridSearchCV, RandomizedSearchCV] = None,
    n_jobs: typing.Union[None, int] = None,
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
        n_jobs=None,  # n_jobs,
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
            return numpy.unique(self._cluster_labels_df["cluster_label"].values).shape[
                0
            ]
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

    # def _identify_clusters(
    #     self,
    #     X: typing.Union[numpy.ndarray, pandas.DataFrame],
    # ) -> tuple[pandas.DataFrame, AutoHierarchicalAssociationClustering]:
    #     """
    #     Given a dataset containing features (columns) and corresponding \
    #     observations (rows) identify highly associated/correlated features by \
    #     grouping these into clusters.

    #     Args:
    #         X: An input dataset containing features (columns) and \
    #             corresponding observations (rows).

    #     Raises:
    #         ValueError: If the given input argument 'X' is not type \
    #             'ndarray' or 'DataFrame'.

    #     Returns:
    #         Data pertaining to the best clustering of features.
    #     """
    #     # if isinstance(X, numpy.ndarray):
    #     #     logging.info("Converting input 'ndarray' 'X' to a 'DataFrame'.")
    #     #     _X = pandas.DataFrame(X)
    #     # elif isinstance(X, pandas.DataFrame):
    #     #     _X = X#.copy()
    #     # else:
    #     #     raise TypeError(
    #     #         "The given input argument 'X' is not of type "
    #     #         + "'ndarray' or 'DataFrame'. 'X' is instead "
    #     #         + f"of type {type(X)}."
    #     #     )
    #     # TODO: Drop a feature (dataframe column) if more than 1/3
    #     #       percent of the values in the column are missing
    #     # TODO: Replace NAN values in a column with the mean of
    #     #       of the values of the feature (dataframe column)
    #     # _X = _X.dropna(axis=0, inplace=False)
    #     # _X = X.dropna(axis=0, inplace=False)
    #     (cluster_labels_df, clustering_model,) = _identify_colinear_features(
    #         # df=_X,
    #         df=X,
    #     )
    #     self._cluster_labels_df = cluster_labels_df
    #     return (
    #         cluster_labels_df,
    #         clustering_model,
    #     )


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
            LGBMClassifier,
            LGBMRegressor,
            # RandomForestClassifier,
            # RandomForestRegressor,
        ],
        scoring: str,
        estimator_params: typing.Union[None, dict[str, typing.Any]] = None,
        n_splits: int = DEFAULT_SPLITS,
        n_repeats: int = DEFAULT_REPEATS,
        random_seed: int = utils.DEFAULT_RANDOM_SEED,
        iterations: typing.Union[None, int] = None,
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

        # Check that the given input is valid
        # self._check_vars()

        # Set random seed for reproducibility purposes
        numpy.random.seed(self.random_seed)

        # Public accessible vars associated with the most important features
        # These vars wil eventually be set after a call to 'fit()'
        # self.ranked_differences: typing.Union[None, pandas.DataFrame] = None
        self.selected_features: typing.Union[None, list[str]] = None

        # Internal vars for easy access to data associated with the importance
        # ranking of features
        self._history: pandas.DataFrame = pandas.DataFrame()
        self._feature_selector: typing.Union[None, FeatureSelectionHelper] = None
        # Private class variable for a progress bar that is to be updated
        # TODO: Determine progrss bar type
        self._progress_bar: typing.Union[None, typing.Any] = None

        # Keep a class variable around to store a plotting interface object
        # such that all necessary plotting methods can be accessed through it
        # self._plotting_interface: typing.Union[
        #     None, ShapFirePlottingInterface
        # ] = None

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

        # feature_associations = utils.associations(X=X)

        # # Turn the association matrix X into a dissimilarity matrix
        # _X = 1 - numpy.abs(feature_associations)
        # # Fill the diagonal elements in the matrix with zeros
        # numpy.fill_diagonal(_X.values, 0)
        # # Make sure the matrix is symmetric
        # pairwise_distances = squareform(X=_X, checks=True, force="tovector")

        # Cluster collinear/multicollinear features
        model = AutoHierarchicalAssociationClustering()
        model.fit(X=X)

        # Organize information in a dictionary and then a dataframe
        dict_cluster_labels: dict[int, typing.Any] = {}
        # Extract a list of feature names
        feature_names = X.columns.to_list()
        # Extract all cluster labels
        if model._idx_to_cluster_array is not None:
            cluster_labels = numpy.unique(model._idx_to_cluster_array)
        else:
            raise ValueError(
                "Internal error."
                + "The indexing array '._idx_to_cluster_array' is None."
            )
        # Make a list entry in the 'dict_cluster_labels' dictionary for each
        # possible cluster label
        for i in range(len(cluster_labels)):
            dict_cluster_labels[int(cluster_labels[i])] = []

        # Populate each of the lists associated with a cluster, with
        # feature names corresponding to the features that were placed
        # in those clusters
        for i in range(len(model._idx_to_cluster_array)):
            dict_cluster_labels[model._idx_to_cluster_array[i]].append(
                feature_names[i]
            )
        lst: list[dict[str, typing.Any]] = []
        for label in dict_cluster_labels:
            for feature_name in dict_cluster_labels[label]:
                d = {"cluster_label": label, "feature_name": feature_name}
                lst.append(d)
        df_cluster_labels = pandas.DataFrame(data=lst)
        logging.info(
            "The following cluster labels have been assigned to "
            + f"the corresponding feature names:\n{df_cluster_labels}",
        )

        # Determine feature clustering
        self._feature_selector = FeatureSelectionHelper(
            random_seed=self.random_seed,
        )
        self._feature_selector._cluster_labels_df = df_cluster_labels

        # # (
        # #     cluster_labels_df,
        # #     clustering_model,
        # # ) = self._feature_selector._identify_clusters(X=X_copy)

        # (cluster_labels_df, clustering_model,) = _identify_colinear_features(
        #     # df=_X,
        #     df=X_copy,
        # )
        # self._cluster_labels_df = cluster_labels_df

        if self.iterations is None:
            self.iterations = self._feature_selector.largest_cluster

        # Create progress bar which will be updated continuously to track the
        # progress of the outer cross-validation loop
        # total = self.n_repeats * self.n_splits * self.iterations
        # self._progress_bar = tqdm(
        #     total=total,
        #     unit_scale=True,
        #     ascii=" >=",
        #     bar_format="{desc:<20}{percentage:3.0f}%|{bar:25}{r_bar}",
        #     desc="ShapFire progress",
        # )

        # estimator = self.estimator_class(
        #     random_state=self.random_seed,
        #     objective="binary",
        # ).fit(
        #     X=X,
        #     y=y
        # )
        # print(dir(estimator))
        # print()
        # print(estimator.objective)
        # # quit()
        # print("Hello World 1")
        # shap_values = shap.TreeExplainer(model=estimator)
        # print("Hello World 2")
        # quit()

        # Perform repeated nested Cross-Validation (CV):
        self._outer_cv_loop(X=X, y=y)

        # Close/stop the progress bar
        # self._progress_bar.close()

        # # Calculate normalized SHAP feature importance scores and pick the best
        # # feature from each of the previously found clusters
        # # df = self._calculate_normalized_shap_feature_importance()
        # self.feature_importances = df = (
        #     self._calculate_normalized_shap_feature_importance()
        # )

        # # Extract and organize data associated with each tested feature subset
        # data_dict = self._reorganize_feature_importance_values(df=df)

        # # TODO: Adjust feature selection strategy!
        # # ndf = self._calculate_ranked_differences(data_dict=data_dict)
        # # self._discard_unimportant_feautres(df=ndf)
        # feature_ranking_df = self._pick_top_k_from_clusters(
        #     df=df,
        #     top_k=None,
        # )

        # # TODO:
        # # Select all features from each cluster
        # selected_features = feature_ranking_df.groupby(by=["cluster"]).head(1)

        # # Discard features with 0 importance. A selected feature should
        # # not have 0 importance!
        # selected_features = selected_features[
        #     selected_features["normalized_feature_importance"] > 0
        # ]

        # # Further filtering based on a cutoff value
        # cut_value = self._find_cutoff(df=selected_features)
        # print("Normalized SHAP importance cutoff value: ", cut_value)
        # selected_features = selected_features[
        #     selected_features["normalized_feature_importance"] >= cut_value
        # ]

        # selected_feature_names = [v[0] for v in selected_features.index.values]
        # self.selected_features = selected_feature_names
        # return self
        return None

    # def transform(
    #     self, X: typing.Union[numpy.ndarray, pandas.DataFrame]
    # ) -> typing.Union[numpy.ndarray, pandas.DataFrame]:
    #     """
    #     Reduce the input dataset X containing features (columns) and
    #     corresponding observations (rows), to only the columns of features
    #     selected by ShapFire.

    #     Args:
    #         X: The original input dataset that the ShapFire method was applied \
    #             to. The dataset is assumed to contain features (columns) and \
    #             corresponding observations (rows).

    #     Raises:
    #         ValueError: If the :meth:`fit` method has not yet been called.

    #     Returns:
    #         A reduced dataset that only contains the most important features \
    #             (columns).
    #     """
    #     if self.selected_features is not None:
    #         return X[self.selected_features]
    #     else:
    #         raise ValueError(
    #             "Method '.fit(X, y)' has not yet been called. "
    #             + "Simply call method '.fit_transform(X, y)' or call "
    #             + "'.fit(X, y)' before calling '.transform(X)'."
    #         )

    # def fit_transform(
    #     self,
    #     X: typing.Union[numpy.ndarray, pandas.DataFrame],
    #     y: typing.Union[numpy.ndarray, pandas.DataFrame],
    # ) -> typing.Union[numpy.ndarray, pandas.DataFrame]:
    #     """
    #     Perform SHAP feature importance rank ensembling for the purpose of
    #     selecting the features that are the most important. Subsequently, reduce
    #     the input dataset 'X' to only the columns of the selected features.

    #     Args:
    #         X: An input dataset that the ShapFire method should be applied to. \
    #             The dataset is assumed to contain features (columns) and \
    #             corresponding observations (rows).
    #         y: The samples associated with the target variable of the dataset.

    #     Returns:
    #         A reduced dataset that only contains the data associated with the \
    #             most important features.
    #     """
    #     self.fit(X=X, y=y)
    #     return self.transform(X=X)

    # def plot_ranking(
    #     self,
    #     groupby: str = "cluster",
    #     rcParams: typing.Union[None, dict[str, str]] = None,
    #     figsize: typing.Union[None, tuple[float, float]] = None,
    #     fontsize: int = 10,
    #     with_text: bool = True,
    #     with_overlay: bool = True,
    #     ax: typing.Union[None, Axes] = None,
    # ) -> tuple[Figure, Axes]:
    #     """
    #     Plot the feature importance scores associated with each feature. The
    #     features will be ordered in the figure from best to worst and possibly
    #     according to which cluster they each belong to.

    #     Args:
    #         groupby: A string value indicating how the feature importance \
    #             ranking should be displayed in a figure. If the option \
    #             'cluster' is chosen, then the features are grouped and \
    #             shown in the figure based on their assigned cluster and \
    #             according to the importance rank of the best feautre in the \
    #             cluster. If 'feature' is chosen, then the features are \
    #             shown in the figure purely according to their global rank \
    #             without any consideration to what cluster each features are a \
    #             part of.
    #         figsize: The width and height of the figure in inches. Defaults to \
    #             None.
    #         fontsize: The size of the font present in the figure. Defaults to \
    #             10.
    #         with_text: If input argument :code:`groupby` is set to \
    #             'cluster', then :code:`with_text` determines whether \
    #             features that have been grouped in the figure by the cluster \
    #             they each belong to, should also be annotated with a text \
    #             label. Defaults to True.
    #         with_overlay: Depending on whether :code:`groupby` is set to \
    #             'cluster' or 'feature', groups of features or individual \
    #             features are assigned a gray-scale overlay creating a visual \
    #             grouping / delimitation of features. Defaults to True.
    #         ax: A Matplotlib Axes object. Defaults to None.

    #     Returns:
    #         A Matplotlib Figure and Axes object.
    #     """
    #     if self._plotting_interface is None:
    #         self._plotting_interface = ShapFirePlottingInterface(shapfire=self)
    #         # TODO: If fit is called again, then self._plotting_interface should
    #         #       be set to None.
    #     return self._plotting_interface.plot_ranking(
    #         groupby=groupby,
    #         rcParams=rcParams,
    #         figsize=figsize,
    #         fontsize=fontsize,
    #         with_text=with_text,
    #         with_overlay=with_overlay,
    #         ax=ax,
    #     )

    # def plot_importance(
    #     self,
    #     plot_type: str = "stripplot",
    #     groupby: str = "cluster",
    #     rcParams: typing.Union[None, dict[str, str]] = None,
    #     figsize: typing.Union[None, tuple[float, float]] = None,
    #     fontsize: int = 10,
    #     with_text: bool = True,
    #     with_overlay: bool = True,
    #     ax: typing.Union[None, Axes] = None,
    # ) -> tuple[Figure, Axes]:
    #     """
    #     Plot the normalized SHAP feature importance scores associated with each
    #     feature. The features will be ordered in the figure from best to worst
    #     and possibly according to which cluster they each belong to.

    #     Args:
    #         plot_type: An argument to control how the normalized SHAP feature \
    #             importance scores should be displayed. Defaults to 'stripplot'.
    #         groupby: A string value indicating how the feature importance \
    #             ranking should be displayed in a figure. If the option \
    #             'cluster' is chosen, then the features are grouped and \
    #             shown in the figure based on their assigned cluster and \
    #             according to the importance rank of the best feautre in the \
    #             cluster. If 'feature' is chosen, then the features are \
    #             shown in the figure purely according to their global rank \
    #             without any consideration to what cluster each features are a \
    #             part of.
    #         figsize: The width and height of the figure in inches. Defaults to \
    #             None.
    #         fontsize: The size of the font present in the figure. Defaults to \
    #             10.
    #         with_text: If input argument :code:`groupby` is set to \
    #             'cluster', then :code:`with_text` determines whether \
    #             features that have been grouped in the figure by the cluster \
    #             they each belong to, should also be annotated with a text \
    #             label. Defaults to True.
    #         with_overlay: Depending on whether :code:`groupby` is set to \
    #             'cluster' or 'feature', groups of features or individual \
    #             features are assigned a gray-scale overlay creating a visual \
    #             grouping / delimitation of features. Defaults to True.
    #         ax: A Matplotlib Axes object. Defaults to None.

    #     Returns:
    #         A Matplotlib Figure and Axes object.
    #     """
    #     if self._plotting_interface is None:
    #         self._plotting_interface = ShapFirePlottingInterface(shapfire=self)
    #         # TODO: If fit is called again, then self._plotting_interface should
    #         #       be set to None.
    #     return self._plotting_interface.plot_importance(
    #         plot_type=plot_type,
    #         groupby=groupby,
    #         rcParams=rcParams,
    #         figsize=figsize,
    #         fontsize=fontsize,
    #         with_text=with_text,
    #         with_overlay=with_overlay,
    #         ax=ax,
    #     )

    # def _check_vars(self) -> None:
    #     _check_scoring_function(
    #         scoring=self.scoring, estimator_class=self.estimator_class
    #     )
    #     if isinstance(self.scoring, str):
    #         self.scoring = self.scoring.strip().lower()
    #     _check_cv_params(n_splits=self.n_splits, n_repeats=self.n_repeats)
    #     if self.iterations is not None:
    #         if self.iterations < 1:
    #             raise ValueError(
    #                 "The given input argument 'iterations' can not be" "less than 1."
    #             )

    # def _calculate_stats(self, true_values, predicted_values):
    #     sensitivity = recall_score(true_values, predicted_values)
    #     specificity = recall_score(
    #         numpy.logical_not(true_values),
    #         numpy.logical_not(predicted_values),
    #     )

    # TODO: Make public and not part of class
    # def _get_score(
    #     self,
    #     estimator: typing.Union[
    #         LGBMClassifier,
    #         LGBMRegressor,
    #         # RandomForestClassifier,
    #         # RandomForestRegressor,
    #     ],
    #     X_test: pandas.DataFrame,
    #     y_test: pandas.Series,
    # ) -> typing.Union[None, dict[str, typing.Any]]:
    #     """
    #     Retrieve the performance score of an estimator on a given test set.

    #     Args:
    #         estimator: A LightGBM estimator from Microsoft's LightGBM \
    #             gradient boosting decision tree framework. The estimator can \
    #             either be a classifier or a regressor. The estimator is \
    #             assumed to have been trained on a training dataset and \
    #             should be evaluated on a test dataset.
    #         X_test: A test dataset.
    #         y_test: The samples associated with the target variable of the \
    #             test dataset.

    #     Raises:
    #         ValueError: If the estimator can not be identified as being a \
    #             classifier or regressor.

    #     Returns:
    #         Returns a dictionary with a performance score and possibly \
    #         additional data pertaining to a certain type of performance score.
    #     """
    #     dict_ = {}
    #     if is_classifier(self.estimator_class):
    #         # Handle special scoring functions where additional data, beyond
    #         # just a score,  needs to be saved and passed on
    #         if self.scoring == "roc_auc":
    #             fpr, tpr, roc_auc = get_roc_auc_statistics(
    #                 estimator=estimator,
    #                 X_test=X_test,
    #                 y_test=y_test,
    #             )
    #             dict_["fpr"] = fpr
    #             dict_["tpr"] = tpr
    #             dict_["roc_auc"] = roc_auc

    #             # # Get confusion matrices
    #             # conf_matrices = get_conf_matrices(
    #             #     estimator=estimator,
    #             #     X_test=X_test,
    #             #     y_test=y_test,
    #             # )
    #             # dict_["conf_matrices"] = conf_matrices
    #             return dict_
    #         else:
    #             raise ValueError("TODO: Not yet implemented!")
    #     elif is_regressor(self.estimator_class):
    #         raise ValueError("TODO: Not yet implemented!")
    #     else:
    #         raise ValueError(
    #             "It could not be determined whether the given "
    #             + f"'estimator': {estimator} is a classifier or a regressor."
    #         )

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
        feature_clusters = list(self._feature_selector.feature_clusters)  # type: ignore
        cs = ClusterSampler(feature_clusters=feature_clusters)

        for _ in range(self.iterations):  # type: ignore
            selected_features = cs.sample_feature_subset()
            for i, (train_ix, test_ix) in enumerate(cv.split(X=X, y=y)):
                X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
                y_train, y_test = y.values[train_ix], y.values[test_ix]

                _X_train, _y_train = X_train[selected_features], y_train
                estimator = self.estimator_class(
                    random_state=self.random_seed,
                    n_jobs=None,
                ).fit(
                    X=_X_train,
                    y=_y_train,
                )

                quit()

                # Retrieve SHAP values on outer loop CV test set using
                # best estimator refitted on inner loop CV training + test set
                # Note: Depending on shap lib. might have to be changed:
                # shap_values = shap.TreeExplainer(estimator).shap_values(
                #     X_test[selected_features]
                # )
                # 
                # explainer = shap.TreeExplainer(estimator)
                # shap_values = explainer.shap_values(X_test[selected_features])

                # Note : Depending on shap lib. might have to be changed:
                values = numpy.abs(shap_values).mean(axis=0)

                feature_importances = pandas.DataFrame(
                    list(zip(selected_features, values)),
                    columns=["feature_name", "feature_importance"],
                )

                feature_importances.sort_values(
                    by=["feature_importance"],
                    ascending=False,
                    inplace=True,
                )
                score: typing.Union[None, dict[str, typing.Any]] = self._get_score(
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
                # self._progress_bar.update(1)  # type: ignore

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

    # def _calculate_normalized_shap_feature_importance(self) -> pandas.DataFrame:
    #     """
    #     Calculate and organize the normalized SHAP feature importance each test
    #     fold in the cross-validation loop. Normalizing SHAP feature importance
    #     scores makes it possible to compare and aggregate results across
    #     different folds if necessary.

    #     Raises:
    #         ValueError: If the internal class variable  '._history' is None.
    #         ValueError: If the internal class variable '._feature_selector' \
    #             is None.
    #         ValueError: If the internal class variable \
    #             '._feature_selector._df_cluster_labels' is None.
    #         ValueError: If a certain column name is not contained in the \
    #             internally used '._history' pandas dataframe.

    #     Returns:
    #         A pandas dataframe that contains normalized SHAP feature importance
    #         values associated with each feature in a tested feature subset.
    #     """
    #     # Validate and check necessary data before proceeding
    #     if self._history is None:
    #         raise ValueError(
    #             "Internal error. The internal class variable "
    #             + "'._history' is None. This should not happend if "
    #             + "the method is called via the '.fit(X, y)' method."
    #         )
    #     if self._feature_selector is None:
    #         raise ValueError(
    #             "Internal error. The internal class variable "
    #             + "'._feature_selector' is None. This should not happend if "
    #             + "the method is called via the '.fit(X, y)' method."
    #         )
    #     if self._feature_selector._cluster_labels_df is None:
    #         raise ValueError(
    #             "Internal error. The internal class variable "
    #             + "'._feature_selector._df_cluster_labels ' is None. This "
    #             + "should not happend if the method is called via the "
    #             + "'.fit(X, y)' method."
    #         )

    #     # Verify that all required data is contained in 'self._history'
    #     for column_name in self._HISTORY_REQUIRED_FIELDS:
    #         if column_name not in self._history.columns:
    #             raise ValueError(
    #                 f"The column name {column_name} is required but is "
    #                 + "not contained in the internally used "
    #                 + "'._history' pandas dataframe."
    #             )
    #     folds: int = self._history.shape[0]
    #     arr = []
    #     for i in range(folds):
    #         df_fold = (
    #             self._history["feature_importances"].iloc[i].reset_index(drop=True)
    #         )
    #         score: float = self._history["score"].iloc[i][self.scoring]

    #         # Sum feature importance value such that we can compute a
    #         # normalized feature importance value that lies in the range
    #         # [0, 1]. This makes it possible to then aggregate and compare
    #         # scores across differrent trained models.
    #         total = df_fold["feature_importance"].sum()

    #         # Create new column with normalized feature importance scores
    #         df_fold["normalized_feature_importance"] = (
    #             df_fold["feature_importance"] / total
    #         )

    #         # Enumerate CV folds from 1...
    #         df_fold.index = df_fold.index + 1
    #         for index, row in df_fold.iterrows():
    #             d = {
    #                 "test_fold": i + 1,
    #                 # Set the feature name
    #                 "feature_name": row["feature_name"],
    #                 # Set the normalized feature importance score calculated
    #                 # based on the outer loop CV test fold
    #                 "normalized_feature_importance": row[
    #                     "normalized_feature_importance"
    #                 ],
    #                 # Set the rank of the feature. The rank is based on the
    #                 # computed 'normalized_feature_importance'
    #                 "feature_rank": index,
    #                 # Set the performance score that was calculated based on
    #                 # the outer loop CV test fold
    #                 "score": score,
    #                 # Retrieve the cluster that the feature belongs to
    #                 "cluster": self._feature_selector._cluster_labels_df[
    #                     self._feature_selector._cluster_labels_df["feature_name"]
    #                     == row["feature_name"]
    #                 ]["cluster_label"].iat[0],
    #             }
    #             arr.append(d)
    #     return pandas.DataFrame(data=arr)

    # def _pick_top_k_from_clusters(
    #     self,
    #     df: pandas.DataFrame,
    #     top_k: typing.Union[None, int] = None,
    # ) -> pandas.DataFrame:
    #     """
    #     The method picks the top k best features, ranked by their normalized
    #     SHAP importance score, from each cluster of highly associated/correlated
    #     features.

    #     Args:
    #         df: A dataframe containing normalized SHAP feature importance \
    #             scores that can be used for ranking the importance of the \
    #             different features.
    #         top_k: The numer of features to pick from each cluster of \
    #             features. Defaults to None.

    #     Raises:
    #         TypeError: If the 'df' input argument is not a pandas dataframe.
    #         ValueError: If the 'top_k' input argument is not an integer value.

    #     Returns:
    #         A dataframe reduced to the top k features from each cluster of \
    #         highly associated/correlated features, ranked by their associated \
    #         normalized SHAP feature importance.
    #     """
    #     REQUIRED_FIELDS = [
    #         "feature_name",
    #         "cluster",
    #         "normalized_feature_importance",
    #     ]
    #     # Validate input arguments before proceeding
    #     if not isinstance(df, pandas.DataFrame):
    #         raise TypeError(
    #             "The internally passed input argument 'df' is not of type "
    #             + f"'DataFrame'. 'df' is instead of type {type(df)}."
    #         )
    #     else:
    #         # Verify that all required data is contained in input argument 'df'
    #         for column_name in REQUIRED_FIELDS:
    #             if column_name not in list(df.columns):
    #                 raise ValueError(
    #                     f"The column name {column_name} is required but is "
    #                     + "not contained in the internally passed input "
    #                     + "argument 'df' pandas dataframe."
    #                 )

    #     # Extract necessary data
    #     # TODO: Maybe make it possible to choose between agg("median") and
    #     #       agg("mean")?
    #     print(df.dtypes)

    #     gdf = df[REQUIRED_FIELDS].groupby(by=["feature_name", "cluster"])
    #     for item in gdf:
    #         print(item)

    #     _df = (
    #         df[REQUIRED_FIELDS]
    #         .groupby(by=["feature_name", "cluster"])
    #         .agg({"normalized_feature_importance": "median"})
    #         .sort_values(
    #             by=["normalized_feature_importance"],
    #             ascending=False,
    #         )
    #         .groupby(by=["cluster"])
    #     )
    #     # if top_k is not None:
    #     #     # Return the top k best ranked features from each cluster
    #     #     return _df.head(top_k)
    #     # else:
    #     #     # Return all features from each cluster
    #     #     return _df.head(numpy.inf)

    #     if top_k is not None:
    #         # Return the top k best ranked features from each cluster
    #         return _df.head(top_k)
    #     else:
    #         # Return all features from each cluster
    #         return _df.head(numpy.inf)

    # def _find_cutoff(self, df, relative_change=0.1):
    #     ndf = (
    #         df.groupby(level=0)
    #         .apply(
    #             # max
    #             numpy.maximum.reduce
    #         )
    #         .sort_values(
    #             by="normalized_feature_importance",
    #             ascending=False,
    #         )
    #     )

    #     values = ndf.values.flatten()
    #     features = ndf.index.values.flatten()
    #     ys = []
    #     for i in range(1, len(values) + 1):
    #         sum1 = values[:i]
    #         sum2 = values[i:]
    #         norm_sum1 = numpy.sum(sum1)
    #         norm_sum2 = numpy.sum(sum2)
    #         # Interpretation is: how much more does remaining feature contributions
    #         # "norm_sum2" explain compared to current total contributions "norm_sum1"
    #         # calculated from the i - n first features
    #         proportion = (norm_sum2 / norm_sum1) * 100
    #         ys.append(proportion)
    #     final_df = pandas.DataFrame(data=ys, columns=["proportions"])
    #     final_df.index = features
    #     # print("Final df: ")
    #     # print(final_df)
    #     cut_feature = final_df[final_df["proportions"] >= relative_change].index[-1]
    #     temp_df = df.droplevel(1)
    #     cut_value = temp_df[temp_df.index == cut_feature][
    #         "normalized_feature_importance"
    #     ].iloc[0]
    #     return cut_value

    # def _reorganize_feature_importance_values(
    #     self, df: pandas.DataFrame
    # ) -> dict[str, pandas.DataFrame]:
    #     # Organize data per tested feature subset
    #     fsc = FeatureSubsetCollection()
    #     for _, _df in df.groupby("test_fold"):
    #         reduced_df = _df[
    #             [
    #                 "test_fold",
    #                 "feature_name",
    #                 "normalized_feature_importance",
    #             ]
    #         ]
    #         pivot_df = reduced_df.pivot(
    #             index=["test_fold"],
    #             columns=["feature_name"],
    #             values=["normalized_feature_importance"],
    #         )
    #         pivot_df = pivot_df["normalized_feature_importance"].reset_index(drop=True)
    #         pivot_df.columns.name = None
    #         names = list(pivot_df.columns)
    #         feature_names = sorted(names)
    #         key = "-".join(feature_names)
    #         fsc._add_entries(key, pivot_df)
    #     return fsc._data_dict

    # def _calculate_ranked_differences(
    #     self, data_dict: dict[str, pandas.DataFrame]
    # ) -> pandas.DataFrame:
    #     # For each evaluated subset of features calculate the ranked differences
    #     # between rankings obtained from SHAP values associated with the feature
    #     # subsets and reference vectors produced based on the same data through
    #     # a data fusion method
    #     evaluated_feature_subsets = []
    #     for key in data_dict:
    #         feature_importance_values = data_dict[key]
    #         if feature_importance_values is not None:
    #             ranked_differences = RankedDifferences(
    #                 reference=self.reference, ascending=False
    #             ).fit(feature_importance_values)
    #             ranked_differences = ranked_differences.to_dict()
    #             ranked_differences[
    #                 "nsamples"
    #             ] = feature_importance_values.shape[0]
    #             evaluated_feature_subsets.append(ranked_differences)
    #     df = pandas.DataFrame(data=evaluated_feature_subsets)
    #     nsamples = df["nsamples"]
    #     ndf = df.drop("nsamples", axis=1)
    #     # Calculate weighted averages
    #     ndf = ndf.multiply(nsamples, axis="rows").sum() / numpy.sum(nsamples)
    #     ndf = ndf.sort_values(ascending=True).to_frame("ranked_difference")
    #     return ndf

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
        # print("THIS ONE IS RUN!")
        self.selected_features = selected_features.index.to_list()

    def _loopkup_cluster_label(self, feature_name: str) -> str:
        # Retrieve the cluster that the given input feature 'feature_name'
        # belongs to
        df = self._feature_selector._cluster_labels_df  # type: ignore
        return df[  # type: ignore
            df["feature_name"] == feature_name  # type: ignore
        ][
            "cluster_label"
        ].iat[0]


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


# class RefitHelper:
#     def __init__(
#         self,
#         feature_names: list[str],
#         estimator_class: typing.Union[
#             LGBMClassifier,
#             LGBMRegressor,
#             # RandomForestClassifier,
#             # RandomForestRegressor,
#         ],
#         scoring: str,
#         estimator_params: typing.Union[None, dict[str, typing.Any]],
#         n_splits: int = DEFAULT_SPLITS,
#         n_repeats: int = DEFAULT_REPEATS,
#         random_seed: int = utils.DEFAULT_RANDOM_SEED,
#     ) -> None:
#         """
#         Args:
#             feature_names: A list of selected features.
#             estimator_class: The scikit-learn or Microsoft LightGBM \
#                 tree-based estimator to use. The estimator can either be a \
#                 classifier or a regressor.
#             scoring: The specification of a scoring function to use for \
#                 model-evaluation, i.e., a function that can be used for \
#                 assessing the prediction error of a trained model given a test \
#                 set.
#             estimator_params: The estimator hyperparameters and corresponding \
#                 values to search or directly use. If only a single value for \
#                 each hyperparameter is provided then only cross-validation \
#                 will be performed and no hyperparameter search will be \
#                 performed. Defaults to None.
#             n_splits: The number of folds to generate in the outer loop \
#                 of a nested cross-validation. Defaults to \
#                     :const:`shapfire.shapfire.DEFAULT_SPLITS`.
#             n_repeats: The number of new folds that should be generated \
#                 in the outer loop of a nested cross-validation. Defaults to \
#                     :const:`shapfire.shapfire.DEFAULT_REPEATS`.
#             random_seed: The random seed to use for reproducibility purposes. \
#                 Defaults to :const:`shapfire.utils.DEFAULT_RANDOM_SEED`.

#         Attributes:
#             history: A class attribute and pandas dataframe that contains the
#                 performance score (and possibly other data) associated with each
#                 test fold in a repeated corss-validation.
#         """
#         # Class vars corresponding to input args
#         self.estimator_class = estimator_class
#         self.scoring = scoring
#         self.estimator_params = estimator_params
#         self.n_splits = n_splits
#         self.n_repeats = n_repeats
#         self.feature_names = feature_names
#         self.random_seed = random_seed

#         # Check that the given input is valid
#         # self._check_vars()

#         # Set random seed for reproducibility purposes
#         numpy.random.seed(self.random_seed)

#         # Public accessible vars associated with the most important features
#         # These vars wil eventually be set after a call to 'fit()'
#         self.history = pandas.DataFrame()

#     def fit(self, X: pandas.DataFrame, y: pandas.Series) -> "RefitHelper":
#         history: list[dict[str, typing.Any]] = []
#         repeat_number = 1

#         cv = get_kfold_cross_validator(
#             estimator_class=self.estimator_class,
#             n_repeats=self.n_repeats,
#             n_splits=self.n_splits,
#         )

#         for i, (train_ix, test_ix) in enumerate(cv.split(X=X, y=y)):
#             X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
#             y_train, y_test = y.values[train_ix], y.values[test_ix]

#             _X_train, _y_train = X_train[self.feature_names], y_train
#             estimator = self.estimator_class(
#                 random_state=self.random_seed,
#                 **self.estimator_params,
#             ).fit(
#                 X=_X_train,
#                 y=_y_train,  # .ravel(),
#             )

#             score: typing.Union[None, dict[str, typing.Any]] = self._get_score(
#                 estimator=estimator,
#                 X_test=X_test[self.feature_names],
#                 y_test=y_test,
#             )

#             if score is None:
#                 raise NotImplementedError(
#                     f"The scorer '{self.scoring}' has not yet been "
#                     + "implemented for use with ShapFire."
#                 )
#             dict_ = {
#                 # 'score' a dictionary that contains data pertaining to
#                 # the estimate of the performance on the outer loop CV test
#                 # set using a certain scoring measure specified by
#                 # 'self.scoring'.
#                 "score": score,
#                 "repeat_number": repeat_number,
#             }
#             history.append(dict_)
#             if ((i + 1) % self.n_splits) == 0:
#                 repeat_number += 1

#         _history = pandas.DataFrame(data=history)
#         # If the current ShapFire object already has a 'self.history'
#         # defined then reset the dataframe so data does not accumulate
#         if self.history is not None:
#             self.history = pandas.DataFrame()
#         self.history = pandas.concat(
#             [
#                 self.history.reset_index(drop=True),
#                 _history.reset_index(drop=True),
#             ],
#             ignore_index=True,
#             join="outer",
#             axis=0,
#         )
#         return self

#     def _get_score(
#         self,
#         estimator: typing.Union[
#             LGBMClassifier,
#             LGBMRegressor,
#             # RandomForestClassifier,
#             # RandomForestRegressor,
#         ],
#         X_test: pandas.DataFrame,
#         y_test: pandas.Series,
#     ) -> typing.Union[None, dict[str, typing.Any]]:
#         """
#         Retrieve the performance score of an estimator on a given test set.

#         Args:
#             estimator: A LightGBM estimator from Microsoft's LightGBM \
#                 gradient boosting decision tree framework. The estimator can \
#                 either be a classifier or a regressor. The estimator is \
#                 assumed to have been trained on a training dataset and \
#                 should be evaluated on a test dataset.
#             X_test: A test dataset.
#             y_test: The samples associated with the target variable of the \
#                 test dataset.

#         Raises:
#             ValueError: If the estimator can not be identified as being a \
#                 classifier or regressor.

#         Returns:
#             Returns a dictionary with a performance score and possibly \
#             additional data pertaining to a certain type of performance score.
#         """
#         dict_ = {}
#         if is_classifier(self.estimator_class):
#             # Handle special scoring functions where additional data, beyond
#             # just a score, needs to be saved and passed on
#             if self.scoring == "roc_auc":
#                 fpr, tpr, roc_auc = get_roc_auc_statistics(
#                     estimator=estimator,
#                     X_test=X_test,
#                     y_test=y_test,
#                 )
#                 dict_["fpr"] = fpr
#                 dict_["tpr"] = tpr
#                 dict_["roc_auc"] = roc_auc

#                 # Get confusion matrices
#                 conf_matrices = get_conf_matrices(
#                     estimator=estimator,
#                     X_test=X_test,
#                     y_test=y_test,
#                 )
#                 dict_["conf_matrices"] = conf_matrices

#                 return dict_
#             else:
#                 raise ValueError("TODO: Not yet implemented!")
#         elif is_regressor(self.estimator_class):
#             raise ValueError("TODO: Not yet implemented!")
#         else:
#             raise ValueError(
#                 "It could not be determined whether the given "
#                 + f"'estimator': {estimator} is a classifier or a regressor."
#             )

#     def _check_vars(self):
#         _check_estimator_class(estimator_class=self.estimator_class)
#         _check_scoring_function(
#             estimator_class=self.estimator_class,
#             scoring=self.scoring,
#         )
#         _check_cv_params(n_splits=self.n_splits, n_repeats=self.n_repeats)


# class ShapFirePlottingInterface:
#     def __init__(self, shapfire) -> None:  # type: ignore
#         # Class vars corresponding to input args
#         self.shapfire = shapfire  # noqa

#         # Internal vars for easy access to data associated with the importance
#         # ranking of features
#         self._data: typing.Union[None, dict[str, typing.Any]] = None
#         self._is_jointplot: bool = False
#         self._groupby: typing.Union[None, str] = None

#     # def plot_ranking(
#     #     self,
#     #     groupby: str = "cluster",
#     #     rcParams: typing.Union[None, dict[str, str]] = None,
#     #     figsize: typing.Union[None, tuple[float, float]] = None,
#     #     fontsize: int = 10,
#     #     with_text: bool = True,
#     #     with_overlay: bool = True,
#     #     ax: typing.Union[None, Axes] = None,
#     # ) -> tuple[Figure, Axes]:
#     #     plot_type = "stripplot"
#     #     # Define the default plotting options
#     #     PLOT_IMPORTANCE_OPTIONS: dict[str, typing.Any] = {
#     #         # Do not allow violinplot. The elements will be squished too
#     #         # much and result in an awful representation of the data
#     #         "stripplot": {
#     #             "func": sns.stripplot,
#     #             "xargs": {"dodge": True, "alpha": 1.0, "ax": ax, "marker": "o"},
#     #         },
#     #     }

#     #     # Validate given input arguments
#     #     if not isinstance(groupby, str):
#     #         raise TypeError(
#     #             "The given input argument 'groupby' should be of type "
#     #             + f"'str' but an argument of type '{type(groupby)}' was given."
#     #         )
#     #     else:
#     #         GROUBPBY_OPTIONS = ["feature", "cluster"]
#     #         if not groupby.strip().lower() in GROUBPBY_OPTIONS:
#     #             raise ValueError(
#     #                 "The given input argument 'groupby' should be one "
#     #                 + f"of the following options: {', '.join(GROUBPBY_OPTIONS)}"
#     #                 + f" but an argument '{groupby}' was given."
#     #             )
#     #     if ax is None:
#     #         # No axis was passed as function input argument. Thus create a new
#     #         # axis object
#     #         fig, ax = plt.subplots(nrows=1, ncols=1)
#     #     else:
#     #         # Get figure from the Axes object so we can subsequently apply
#     #         # styling to it
#     #         fig = ax.get_figure()

#     #     # Apply styling to the plot elements
#     #     _apply_default_styling(rcParams)

#     #     # Prepare the appropriate data for plotting
#     #     _groupby = groupby.strip().lower()
#     #     if self._data is None or _groupby != self._groupby:
#     #         self._data = self._prepare_data(groupby=_groupby)
#     #         self._groupby = _groupby

#     #     # Unpack all necessary data for plotting
#     #     df = self._data["df"]
#     #     feature_ordering = self._data["feature_ordering"]
#     #     feature_colors = self._data["feature_colors"]
#     #     cluster_labels = self._data["cluster_labels"]

#     #     # Determine the searborn function to use for plotting and set function
#     #     # arguments that should be passed to the plotting function
#     #     args = {
#     #         "x": "ranked_difference",
#     #         "y": df.index,
#     #         "data": df,
#     #     }
#     #     plotting_function = PLOT_IMPORTANCE_OPTIONS[plot_type]["func"]
#     #     args.update(PLOT_IMPORTANCE_OPTIONS[plot_type]["xargs"])
#     #     ax = plotting_function(
#     #         order=feature_ordering,
#     #         palette=list(feature_colors.values()),
#     #         **args,
#     #     )
#     #     sns.despine(
#     #         ax=ax,
#     #         top=True,
#     #         right=True,
#     #         left=True,
#     #         bottom=True,
#     #         offset=None,
#     #         trim=False,
#     #     )

#     #     x_max = df["ranked_difference"].max()
#     #     df["ranked_difference"].min()

#     #     # Add additional plot overlays depending on how features should be
#     #     # grouped and displayed in the plot
#     #     if groupby.strip().lower() == "cluster":
#     #         if with_text is True or with_overlay is True:
#     #             # Add two alternating gray-scale colors for grouping features
#     #             # based on the cluster they each belong to. Also, add text
#     #             # information about the cluster each feature belongs to
#     #             self._add_cluster_overlays(
#     #                 ax=ax,
#     #                 cluster_labels=cluster_labels,
#     #                 with_text=with_text,
#     #                 with_overlay=with_overlay,
#     #                 x_offset=x_max * 1.075,
#     #             )
#     #     elif groupby.strip().lower() == "feature":
#     #         if with_overlay is True:
#     #             # Add two alternating gray-scale colors for better seperation
#     #             # of the plotted data. By default do not add text information
#     #             # about the clusters each feature belong to. For this purpose,
#     #             # the groupby = "cluster" should be chosen
#     #             self._add_feature_overlays(ax=ax, cluster_labels=cluster_labels)
#     #     else:
#     #         raise ValueError(
#     #             "The given input argument 'groupby' should have value "
#     #             + f"'feature' or 'cluster' but value '{groupby}' was given."
#     #         )

#     #     # Indicate the cut-off threshold
#     #     ax.axvline(
#     #         self.shapfire.threshold_finder.lower_threshold,
#     #         linestyle="--",
#     #         lw=2.0,
#     #         color=MAIN_COLOR_PALETTE["tertiary"],
#     #     )

#     #     # Add a legend to the figure indicating which feautres have been
#     #     # selected and which have been rejected
#     #     custom_lines = [
#     #         Line2D(
#     #             [0],
#     #             [0],
#     #             color=MAIN_COLOR_PALETTE["tertiary"],
#     #             lw=2.25,
#     #             linestyle="--",
#     #         ),
#     #         Line2D(
#     #             [0],
#     #             [0],
#     #             color=MAIN_COLOR_PALETTE["selected"],
#     #             lw=2.25,
#     #         ),
#     #         Line2D(
#     #             [0],
#     #             [0],
#     #             color=MAIN_COLOR_PALETTE["rejected"],
#     #             lw=2.25,
#     #         ),
#     #     ]
#     #     ax.legend(
#     #         custom_lines,
#     #         ["Threshold", "Selected", "Rejected"],
#     #         loc="upper right",
#     #         fontsize=fontsize + 1,
#     #         handlelength=2.75,
#     #     )

#     #     # Make changes related to figure size, title, x-axis labels + ticks,
#     #     # y-axis labels + and ticks, etc.
#     #     ax.set_title(
#     #         "Importance ranking & selected features",
#     #         fontsize=fontsize + 1,
#     #         pad=20,
#     #     )
#     #     ax.set_xlabel("Ranked difference", fontsize=fontsize + 1)
#     #     ax.set_ylabel("Feature name", fontsize=fontsize + 1)

#     #     # Set x and y-axis ticks
#     #     ax.tick_params(axis="both", which="major", labelsize=fontsize)
#     #     ax.tick_params(axis="both", which="minor", labelsize=fontsize)
#     #     ax.set_zorder(1)

#     #     # Set figure size
#     #     fig = _set_figure_size(
#     #         figsize=figsize,
#     #         fig=fig,
#     #         num_all_features=len(numpy.unique(df.index.to_list())),
#     #     )
#     #     return fig, ax

#     def _organize_data_for_plotting(
#         self,
#         feature_ranking_df: pandas.DataFrame,
#         df: pandas.DataFrame,
#         groupby: str,
#     ) -> tuple[pandas.DataFrame, list[str]]:
#         """
#         Organize and structure the results obtained by applying ShapFire such
#         that the results can easily be plotted and displayed in a figure.

#         Args:
#             feature_ranking_df: _description_
#             df: A dataframe containing all the necessary data for visualizing \
#                 the importance ranking of features.
#             groupby: A string value indicating how the feature importance \
#                 ranking should be displayed in a figure. If the option \
#                 'cluster' is chosen, then the features are grouped and shown \
#                 in the figure based on their assigned cluster and according to \
#                 the importance rank of the best feautre in the cluster. If \
#                 'feature' is chosen, then the features are shown in the figure \
#                 purely according to their global rank without any \
#                 consideration to what cluster each features are a part of.

#         Raises:
#             TypeError: If the input argument 'groupby' is not a string.
#             ValueError: If the input argument 'groupby' is not a valid option.

#         Returns:
#             Organized and structured data of ShapFire results that can be \
#             passed on to appropriate plotting methods.
#         """
#         if not isinstance(groupby, str):
#             raise TypeError(
#                 "Function argument 'groupby' should be of type 'str' but "
#                 + f"argument of type {type(groupby)} was given."
#             )
#         else:
#             names = numpy.unique(df["feature_name"]).tolist()
#             template_dict = {n: numpy.nan for n in names}
#             data = []
#             if groupby.strip().lower() == "feature":
#                 indexing = (
#                     df[["feature_name", "normalized_feature_importance"]]
#                     .groupby("feature_name")
#                     .agg("median")
#                     .sort_values(
#                         by=["normalized_feature_importance"],
#                         ascending=True,
#                     )
#                     .index
#                 )
#                 for _name, _df in df[
#                     ["feature_name", "normalized_feature_importance"]
#                 ].groupby("feature_name"):
#                     for _, _row in _df.iterrows():
#                         # dict_ = template_dict.copy()
#                         dict_ = template_dict#.copy()
#                         dict_[_name] = _row["normalized_feature_importance"]
#                         data.append(dict_)
#                 # Return tuple:
#                 # - Data
#                 # - Vertical ordering by feature name according to feature
#                 #   importance rank
#                 new_df = pandas.DataFrame(data=data).reindex(indexing, axis=1)
#                 return new_df, list(reversed(new_df.columns.values))
#             elif groupby.strip().lower() == "cluster":
#                 indexing = feature_ranking_df.index.values
#                 for _name, _df in df[
#                     ["feature_name", "normalized_feature_importance"]
#                 ].groupby("feature_name"):
#                     for _, _row in _df.iterrows():
#                         # dict_ = template_dict.copy()
#                         dict_ = template_dict
#                         dict_[_name] = _row["normalized_feature_importance"]
#                         data.append(dict_)
#                 # Return tuple:
#                 # - Data
#                 # - Vertical ordering by feature name according to feature
#                 #   importance rank and cluster label
#                 new_df = pandas.DataFrame(data=data).reindex(indexing, axis=1)
#                 return new_df, list(new_df.columns.values)
#             else:
#                 raise ValueError(
#                     "The given input argument 'groupby' should have value "
#                     + f"'feature' or 'cluster' but a value '{groupby}' was "
#                     + "given."
#                 )

#     # def _prepare_data(
#     #     self,
#     #     groupby: str,
#     # ) -> dict[str, typing.Any]:
#     #     feature_ranking_df = self.shapfire.ranked_differences

#     #     if groupby.strip().lower() == "cluster":
#     #         _feature_ranking_df = feature_ranking_df.copy()
#     #         cluster_ordering = (
#     #             _feature_ranking_df.groupby(by=["cluster_label"])
#     #             .head(1)["cluster_label"]
#     #             .values
#     #         )
#     #         _feature_ranking_df["cluster_label"] = pandas.Categorical(
#     #             _feature_ranking_df["cluster_label"].values,
#     #             categories=cluster_ordering,
#     #         )
#     #         _feature_ranking_df.sort_values(
#     #             by=["cluster_label", "ranked_difference"],
#     #             ascending=[True, True],
#     #             inplace=True,
#     #         )
#     #     elif groupby.strip().lower() == "feature":
#     #         _feature_ranking_df = feature_ranking_df.copy()
#     #     else:
#     #         raise ValueError(
#     #             "The given input argument 'groupby' should have value "
#     #             + f"'feature' or 'cluster' but value '{groupby}' was given."
#     #         )

#     #     # Extract a list of clusters that each selected feature is associated
#     #     # with
#     #     cluster_labels = _feature_ranking_df["cluster_label"].values.tolist()
#     #     all_feature_names = _feature_ranking_df.index.to_list()
#     #     selected_features = self.shapfire.selected_features

#     #     # Set colors for each selected/rejected feature
#     #     feature_colors = {}
#     #     for feature_name in all_feature_names:
#     #         if feature_name in selected_features:
#     #             feature_colors[feature_name] = MAIN_COLOR_PALETTE["selected"]
#     #         else:
#     #             feature_colors[feature_name] = MAIN_COLOR_PALETTE["rejected"]

#     #     x = self.shapfire.ranked_differences["ranked_difference"]
#     #     y = self.shapfire.ranked_differences.index
#     #     num_all_features = numpy.unique(y).shape[0]

#     #     _y = y.to_frame()
#     #     _y["feature_name"] = pandas.Categorical(
#     #         _y.index,
#     #         all_feature_names,
#     #     )
#     #     _y.sort_values(by=["feature_name"], inplace=True)

#     #     return {
#     #         "df": self.shapfire.ranked_differences,
#     #         "feature_ordering": all_feature_names,
#     #         "feature_colors": feature_colors,
#     #         "selected_features": selected_features,
#     #         "all_feature_names": all_feature_names,
#     #         "cluster_labels": cluster_labels,
#     #         # Other fields...
#     #         "x": x,
#     #         "y": y,
#     #         "_y": _y,
#     #         "num_all_features": num_all_features,
#     #     }

#     def _prepare_data(
#         self,
#         groupby: str,
#     ) -> dict[str, typing.Any]:
#         # Calculate normalized SHAP feature importance scores
#         df = self.shapfire.feature_importances# self.shapfire._calculate_normalized_shap_feature_importance()


#         # Extract feature ranking, i.e., determine the ordering of the
#         # features on the x-axis based on the computed median
#         # 'normalized_feature_importance'
#         feature_ranking_df = self.shapfire._pick_top_k_from_clusters(
#             df=df,
#             top_k=None,
#         )

#         # TEMP!
#         self._temp_feature_ranking_df = feature_ranking_df

#         if groupby.strip().lower() == "cluster":
#             # _feature_ranking_df = feature_ranking_df.copy()
#             _feature_ranking_df = feature_ranking_df
#             cluster_ordering = (
#                 _feature_ranking_df.reset_index(level=["cluster"])
#                 .groupby(by=["cluster"])
#                 .head(1)["cluster"]
#                 .values
#             )
#             _feature_ranking_df = _feature_ranking_df.reset_index(
#                 level=["cluster"]
#             )
#             _feature_ranking_df["cluster"] = pandas.Categorical(
#                 _feature_ranking_df["cluster"].values,
#                 categories=cluster_ordering,
#             )
#             _feature_ranking_df.sort_values(
#                 by=["cluster", "normalized_feature_importance"],
#                 ascending=[True, False],
#                 inplace=True,
#             )
#         elif groupby.strip().lower() == "feature":
#             # _feature_ranking_df = feature_ranking_df.copy()
#             _feature_ranking_df = feature_ranking_df
#             _feature_ranking_df = _feature_ranking_df.reset_index(
#                 level=["cluster"]
#             )
#         else:
#             raise ValueError(
#                 "The given input argument 'groupby' should have value "
#                 + f"'feature' or 'cluster' but value '{groupby}' was given."
#             )

#         # Extract a list of clusters that each selected feature is associated
#         # with
#         cluster_labels = _feature_ranking_df["cluster"].values.tolist()
#         # all_features = feature_ranking_df.reset_index(level=["cluster"])
#         all_feature_names = _feature_ranking_df.index.values

#         # Select best features from each cluster
#         # selected_features = feature_ranking_df.groupby(by=["cluster"]).head(1)

#         # TODO:
#         # Select all features from each cluster
#         selected_features = feature_ranking_df.groupby(by=["cluster"]).head(1)

#         # Discard features with 0 importance. A selected feature should
#         # not have 0 importance!
#         selected_features = selected_features[
#             selected_features["normalized_feature_importance"] > 0
#         ]

#         # Further filtering based on a cutoff value
#         cut_value = self.shapfire._find_cutoff(df=selected_features)
#         print("SHAP importance cutoff value: ", cut_value)
#         selected_features = selected_features[
#             selected_features["normalized_feature_importance"] >=  cut_value
#         ]

#         selected_feature_names = [v[0] for v in selected_features.index.values]

#         plotting_df, feature_ordering = self._organize_data_for_plotting(
#             df=df,
#             feature_ranking_df=_feature_ranking_df,
#             groupby=groupby,
#         )

#         # Set colors for each selected/rejected feature
#         feature_colors = {}
#         for feature_name in all_feature_names:
#             if feature_name in selected_feature_names:
#                 feature_colors[feature_name] = MAIN_COLOR_PALETTE["selected"]
#             else:
#                 feature_colors[feature_name] = MAIN_COLOR_PALETTE["rejected"]

#         # Generate information related to feature subsets
#         # Determine the ordering of the different evaluated feature
#         # subsets on the y-axis based on the score associated with the
#         # feature subset and that has been obtained
#         print()
#         print(df.groupby("test_fold"))
#         print()
#         test_folds = (
#             df.groupby("test_fold")
#             .agg({"score": "median"})
#             .sort_values(by=["score"], ascending=True)
#             .index
#         )

#         x_name = "test_fold_rank"

#         for counter, value in enumerate(test_folds):
#             df.loc[df["test_fold"] == value, x_name] = counter + 1

#         # Extract data necessary for plotting feature subsets
#         x, y = df[x_name], df["feature_name"]
#         num_all_features = numpy.unique(df["feature_name"]).shape[0]
#         num_all_feature_subsets = numpy.unique(df[x_name]).shape[0]
#         best_score = df.loc[df[x_name] == df[x_name].max(), "score"].iat[0]
#         worst_score = df.loc[df[x_name] == df[x_name].min(), "score"].iat[0]

#         _y = y.to_frame()
#         _y["feature_name"] = pandas.Categorical(
#             _y["feature_name"],
#             all_feature_names,
#         )
#         _y.sort_values(by=["feature_name"], inplace=True)

#         return {
#             # Main importance plot fields...
#             "df": df,
#             "feature_ordering": feature_ordering,
#             "feature_colors": feature_colors,
#             "selected_features": selected_features,
#             # "all_features": all_features,
#             "all_features": None,
#             "all_feature_names": all_feature_names,
#             "cluster_labels": cluster_labels,
#             "plotting_df": plotting_df,
#             # Other fields...
#             "x": x,
#             "y": y,
#             "_y": _y,
#             "num_all_features": num_all_features,
#             "num_all_feature_subsets": num_all_feature_subsets,
#             "best_score": best_score,
#             "worst_score": worst_score,
#         }


#     def _add_cluster_overlays(
#         self,
#         ax: Axes,
#         cluster_labels: list[str],
#         fontsize: int = 10,
#         with_text: bool = True,
#         with_overlay: bool = True,
#         x_offset: float = 0,
#     ) -> None:
#         # y-offset. Move text slightly down
#         text_placement_offset = 0.00
#         current_cluster_label = cluster_labels[0]
#         lower_value = -0.5
#         upper_value = 0.5
#         last_index = len(cluster_labels[1:]) + 1

#         # Alpha values associated with the two alternating overlays
#         alphas = [0.05, 0.25]

#         # Values pertaining to first cluster overlay
#         counter0 = 1
#         counter1 = 0
#         alpha = alphas[(counter1 + 1) % 2]

#         # Add overlays by looping over cluster labels associated
#         # with each feature present in the input dataset
#         for i in range(1, last_index):
#             if current_cluster_label != cluster_labels[i]:
#                 if with_overlay is True:
#                     ax.axhline(upper_value, color="black", alpha=0.10)
#                     ax.axhspan(
#                         lower_value,
#                         upper_value,
#                         facecolor=MAIN_COLOR_PALETTE["overlay"],
#                         alpha=alpha,
#                     )
#                 if with_text is True:
#                     ax.text(
#                         x=x_offset,
#                         # Text placement
#                         y=lower_value
#                         + (upper_value - lower_value) / 2
#                         + text_placement_offset,
#                         s=f"Cluster {current_cluster_label}",
#                         fontsize=fontsize,
#                         verticalalignment="center",
#                     )
#                 counter0 = 1
#                 counter1 += 1
#                 alpha = alphas[(counter1 + 1) % 2]
#                 lower_value = upper_value
#                 upper_value += 1.00
#             else:
#                 counter0 += 1
#                 upper_value += 1.00
#             current_cluster_label = cluster_labels[i]
#         if with_overlay is True:
#             ax.axhspan(
#                 lower_value,
#                 upper_value,
#                 facecolor=MAIN_COLOR_PALETTE["overlay"],
#                 alpha=alpha,
#             )
#         if with_text is True:
#             ax.text(
#                 x=x_offset,
#                 # Text placement
#                 y=lower_value
#                 + (upper_value - lower_value) / 2
#                 + text_placement_offset,
#                 s=f"Cluster {current_cluster_label}",
#                 fontsize=fontsize,
#                 verticalalignment="center",
#             )

#     def _add_feature_overlays(
#         self,
#         ax: Axes,
#         cluster_labels: list[str],
#     ) -> None:
#         cluster_labels[0]
#         lower_value = -0.5
#         upper_value = 0.5
#         last_index = len(cluster_labels[1:]) + 1

#         # Opacity values associated with the two alternating overlays
#         alphas = [0.05, 0.25]

#         # Values pertaining to first cluster overlay
#         counter1 = 0
#         alpha = alphas[(counter1 + 1) % 2]

#         # Add overlays
#         for _ in range(1, last_index):
#             ax.axhline(upper_value, color="black", alpha=0.10)
#             ax.axhspan(
#                 lower_value,
#                 upper_value,
#                 facecolor=MAIN_COLOR_PALETTE["overlay"],
#                 alpha=alpha,
#             )
#             counter1 += 1
#             alpha = alphas[(counter1 + 1) % 2]
#             lower_value = upper_value
#             upper_value += 1.00
#         ax.axhspan(
#             lower_value,
#             upper_value,
#             facecolor=MAIN_COLOR_PALETTE["overlay"],
#             alpha=alpha,
#         )


#     def ceil5(self, x: typing.Union[float, int]) -> int:
#         """
#         Given an input value round the value to closest and largest multiple of
#         5.

#         Args:
#             x: A value that is to be rounded.

#         Returns:
#             The input value rounded to the closest and largest multiple
#             of 5.
#         """
#         base = 5
#         return int(base * numpy.ceil(x / base))

#     def plot_importance(
#         self,
#         plot_type: str = "stripplot",
#         groupby: str = "cluster",
#         rcParams: typing.Union[None, dict[str, str]] = None,
#         figsize: typing.Union[None, tuple[float, float]] = None,
#         fontsize: int = 10,
#         with_text: bool = True,
#         with_overlay: bool = True,
#         ax: typing.Union[None, Axes] = None,
#     ) -> tuple[Figure, Axes]:
#         # Define the default plotting options
#         PLOT_IMPORTANCE_OPTIONS: dict[str, typing.Any] = {
#             # Do not allow violinplot. The elements will be squished too
#             # much and result in an awful representation of the data
#             "stripplot": {
#                 "func": sns.stripplot,
#                 "xargs": {"dodge": True, "alpha": 0.66, "ax": ax},
#             },
#             "swarmplot": {
#                 "func": sns.swarmplot,
#                 "xargs": {},
#             },
#             "boxplot": {
#                 "func": sns.boxplot,
#                 "xargs": {
#                     "medianprops": {
#                         "color": "white",
#                         "linewidth": 1.25,
#                     },
#                     "boxprops": {
#                         "linewidth": 0.5,
#                     },
#                     "whiskerprops": {
#                         "linewidth": 1.5,
#                     },
#                     "capprops": {
#                         "linewidth": 1.5,
#                     },
#                 },
#             },
#         }

#         # Validate given input arguments
#         if not isinstance(plot_type, str):
#             raise TypeError(
#                 "The given input argument 'plot_type' should be of type "
#                 + f"'str' but argument of type '{type(plot_type)}' was given."
#             )
#         else:
#             _PLOT_OPTIONS = list(PLOT_IMPORTANCE_OPTIONS.keys())
#             if not plot_type.strip().lower() in _PLOT_OPTIONS:
#                 raise ValueError(
#                     "The given input argument 'plot_type' should be one "
#                     + f"of the following options: {', '.join(_PLOT_OPTIONS)} "
#                     + f" but an argument '{plot_type}' was given."
#                 )

#         if not isinstance(groupby, str):
#             raise TypeError(
#                 "The given input argument 'groupby' should be of type "
#                 + f"'str' but an argument of type '{type(groupby)}' was given."
#             )
#         else:
#             GROUBPBY_OPTIONS = ["feature", "cluster"]
#             if not groupby.strip().lower() in GROUBPBY_OPTIONS:
#                 raise ValueError(
#                     "The given input argument 'plot_type' should be one "
#                     + f"of the following options: {', '.join(_PLOT_OPTIONS)} "
#                     + f" but an argument '{plot_type}' was given."
#                 )
#         if ax is None:
#             # No axis was passed as function input argument. Thus create a new
#             # axis object
#             fig, ax = plt.subplots(nrows=1, ncols=1)
#         else:
#             # Get figure from the Axes object so we can subsequently apply
#             # styling to it
#             fig = ax.get_figure()

#         # Apply styling to the plot elements
#         self._apply_styling(rcParams)


#         # Prepare the appropriate data for plotting
#         _groupby = groupby.strip().lower()
#         if self._data is None or _groupby != self._groupby:
#             self._data = self._prepare_data(groupby=_groupby)
#             self._groupby = _groupby


#         # Unpack all necessary data for plotting
#         df = self._data["df"]
#         feature_ordering = self._data["feature_ordering"]
#         feature_colors = self._data["feature_colors"]
#         cluster_labels = self._data["cluster_labels"]

#         # Determine the searborn function to use for plotting and set function
#         # arguments that should be passed to the plotting function
#         args = {
#             "x": "normalized_feature_importance",
#             "y": "feature_name",
#             "data": df,
#         }

#         plotting_function = PLOT_IMPORTANCE_OPTIONS[plot_type]["func"]
#         args.update(PLOT_IMPORTANCE_OPTIONS[plot_type]["xargs"])
#         ax = plotting_function(
#             order=feature_ordering,
#             palette=list(feature_colors.values()),
#             **args,
#         )
#         # sns.despine(
#         #     ax=ax,
#         #     top=True,
#         #     right=True,
#         #     left=True,
#         #     bottom=True,
#         #     offset=None,
#         #     trim=False,
#         # )

#         quit()


#         x_max = df["normalized_feature_importance"].max().max()
#         df["normalized_feature_importance"].min().min()
#         ax.set_xlim([0.00 - 0.025, x_max + 0.025])
#         # 0.00 - 0.025
#         # x_max + 0.025

#         # Add additional plot overlays depending on how features should be
#         # grouped and displayed in the plot
#         if groupby.strip().lower() == "cluster":
#             if with_text is True or with_overlay is True:
#                 # Add two alternating gray-scale colors for grouping features
#                 # based on the cluster they each belong to. Also, add text
#                 # information about the cluster each feature belongs to
#                 self._add_cluster_overlays(
#                     ax=ax,
#                     cluster_labels=cluster_labels,
#                     with_text=with_text,
#                     with_overlay=with_overlay,
#                     x_offset=x_max * 1.10,
#                 )
#         elif groupby.strip().lower() == "feature":
#             if with_overlay is True:
#                 # Add two alternating gray-scale colors for better seperation
#                 # of the plotted data. By default do not add text information
#                 # about the clusters each feature belong to. For this purpose,
#                 # the groupby = "cluster" should be chosen
#                 self._add_feature_overlays(ax=ax, cluster_labels=cluster_labels)
#         else:
#             raise ValueError(
#                 "The given input argument 'groupby' should have value "
#                 + f"'feature' or 'cluster' but value '{groupby}' was given."
#             )

#         # Add a legend to the figure indicating which feautres have been
#         # selected and which have been rejected
#         custom_lines = [
#             Line2D(
#                 [0],
#                 [0],
#                 color=MAIN_COLOR_PALETTE["selected"],
#                 lw=4.5,
#             ),
#             Line2D(
#                 [0],
#                 [0],
#                 color=MAIN_COLOR_PALETTE["rejected"],
#                 lw=4.5,
#             ),
#         ]
#         ax.legend(
#             custom_lines,
#             ["Selected", "Rejected"],
#             loc="lower right",
#             fontsize=fontsize + 1,
#         )

#         # Make changes related to figure size, title, x-axis labels + ticks,
#         # y-axis labels + and ticks, etc.
#         ax.set_title(
#             "ShapFire importance ranking and selected features",
#             fontsize=fontsize + 1,
#             pad=20,
#         )
#         ax.set_xlabel(
#             "Normalized SHAP feature importance", fontsize=fontsize + 1
#         )
#         # Only display y-axis label if it is plotted alone
#         if self._is_jointplot is False:
#             ax.set_ylabel("Feature name", fontsize=fontsize + 1)
#         else:
#             ax.set_ylabel(None)
#         ax.tick_params(axis="both", which="major", labelsize=fontsize)
#         ax.tick_params(axis="both", which="minor", labelsize=fontsize)
#         ax.set_zorder(1)
#         return fig, ax

#     def plot_evaluated_feature_subsets(
#         self,
#         groupby: str = "cluster",
#         rcParams: typing.Union[None, dict[str, str]] = None,
#         figsize: typing.Union[None, tuple[float, float]] = None,
#         fontsize: int = 10,
#         marker: str = "o",
#         markersize: int = 5,
#         with_text: bool = True,
#         with_overlay: bool = True,
#         axes: typing.Union[None, list[Axes]] = None,
#     ) -> tuple[Figure, list[Axes]]:
#         # Validate given input arguments
#         if isinstance(axes, list):
#             if len(axes) >= 2:
#                 ax0, ax1 = axes[0], axes[1]
#                 fig = ax0.get_figure()
#             else:
#                 raise ValueError(
#                     "The given input argument 'axes' should have length "
#                     + f">= 2 but the given list instead had length {len(axes)}."
#                 )
#         elif axes is None:
#             fig, axes = plt.subplots(
#                 nrows=1,
#                 ncols=2,
#                 sharey=True,
#                 sharex=False,
#                 gridspec_kw={"width_ratios": [5, 1]},
#             )
#             ax0, ax1 = axes[0], axes[1]
#         else:
#             raise ValueError(
#                 "The given input argument 'axes' should be of type 'list' "
#                 + f" or 'None' but is instead of type {type(axes)}."
#             )

#         # Apply styling to the plot elements
#         self._apply_styling(rcParams=rcParams)

#         # Prepare the appropriate data for plotting
#         _groupby = groupby.strip().lower()
#         if self._data is None or _groupby != self._groupby:
#             self._data = self._prepare_data(groupby=_groupby)
#             self._groupby = _groupby

#         # Only adjust the size of the figure here if it is not being plotted
#         # with together with other types of plots in the current figure
#         if self._is_jointplot is False:
#             # fig, figsize = self._set_figure_size(
#             fig, figsize = _set_figure_size(
#                 figsize=figsize,
#                 fig=fig,
#                 num_all_features=len(
#                     numpy.unique(self._data["df"]["feature_name"])
#                 ),
#             )

#         # Unpack all necessary data for plotting
#         all_feature_names = self._data["all_feature_names"]
#         df = self._data["df"]
#         x = self._data["x"]
#         y = self._data["y"]
#         _y = self._data["_y"]
#         self._data["num_all_features"]
#         self._data["num_all_feature_subsets"]
#         best_score = self._data["best_score"]
#         worst_score = self._data["worst_score"]
#         cluster_labels = self._data["cluster_labels"]

#         # Create stripplot
#         sns.stripplot(
#             x=x,
#             y=y,
#             s=markersize,
#             marker=marker,
#             linewidth=1.5,
#             color="black",
#             ec="black",
#             fc="none",
#             jitter=False,
#             order=all_feature_names,
#             ax=ax0,
#         )
#         sns.despine(
#             ax=ax0,
#             top=True,
#             bottom=True,
#             right=True,
#             left=True,
#             offset=None,
#             trim=False,
#         )

#         # Create an associated vertical histogram displaying the number of times
#         # a certain feature has been a part of an evaluated feature subset
#         sns.histplot(
#             y=_y["feature_name"],
#             discrete=True,
#             linewidth=1.5,
#             ax=ax1,
#             color="black",
#             shrink=0.95,
#         )
#         sns.despine(
#             ax=ax1,
#             top=True,
#             bottom=True,
#             right=True,
#             left=True,
#             offset=None,
#             trim=False,
#         )

#         # Extract the histogram max and min values so the axis can be adjusted
#         # accordingly
#         ymax = self.ceil5(df["feature_name"].value_counts().max())
#         ax1.set_xticks([0, ymax])
#         ax1.set_xlim([0, ymax + 1])

#         # Set axis ax1 tick size
#         ax1.tick_params(axis="both", which="major", labelsize=fontsize)
#         ax1.tick_params(axis="both", which="minor", labelsize=fontsize)

#         # Set axis ax0 tick size
#         ax0.tick_params(axis="both", which="major", labelsize=fontsize)
#         ax0.tick_params(axis="both", which="minor", labelsize=fontsize)

#         # Add additional plot overlays depending on how features should be
#         # grouped and displayed in the plot
#         if groupby.strip().lower() == "cluster":
#             if with_text is True or with_overlay is True:
#                 # Add two alternating gray-scale colors for grouping features
#                 # based on the cluster they each belong to. Also, add text
#                 # information about the cluster each feature belongs to
#                 # - Overlays on axis: ax0
#                 self._add_cluster_overlays(
#                     ax=ax0,
#                     cluster_labels=cluster_labels,
#                     # Do not display text. The placement will not wrong!
#                     with_text=False,
#                     with_overlay=with_overlay,
#                 )
#                 # - Overlays on axis: ax1
#                 self._add_cluster_overlays(
#                     ax=ax1,
#                     cluster_labels=cluster_labels,
#                     with_text=with_text,
#                     with_overlay=with_overlay,
#                     x_offset=ymax * 1.10,
#                 )
#         elif groupby.strip().lower() == "feature":
#             if with_overlay is True:
#                 # Add two alternating gray-scale colors for better seperation
#                 # of the plotted data. By default do not add text information
#                 # about the clusters each feature belong to. For this purpose,
#                 # the groupby = "cluster" should be chosen.
#                 # - Overlays on axis: ax0
#                 self._add_feature_overlays(
#                     ax=ax0,
#                     cluster_labels=cluster_labels,
#                 )
#                 # - Overlays on axis: ax1
#                 self._add_feature_overlays(
#                     ax=ax1,
#                     cluster_labels=cluster_labels,
#                 )
#         else:
#             raise ValueError(
#                 "The given input argument 'groupby' should have value "
#                 + f"'feature' or 'cluster' but value '{groupby}' was given."
#             )

#         subsets = 0
#         # Add vertical lines to visually indicate feautre subsets
#         for index, _df in df.groupby("test_fold"):
#             ax0.plot(
#                 [index for _ in ax0.get_yticks()],
#                 ax0.get_yticks(),
#                 color="black",
#                 alpha=0.125,
#             )
#             subsets += 1

#         # Make changes related to figure size, title, x-axis labels + ticks,
#         # y-axis labels + and ticks, etc.

#         # Set plot labels
#         ax0.set_xlabel(
#             "Feature subset ordered by score",
#             fontsize=fontsize + 1,
#         )
#         ax0.set_ylabel("Feature name", fontsize=fontsize + 1)
#         ax0.set_title(
#             "Evaluated feature subsets",
#             fontsize=fontsize + 1,
#             pad=20,
#         )
#         ax1.set_xlabel(
#             "Counts",
#             fontsize=fontsize + 1,
#         )

#         # Add major axis ticks and remove minor axis ticks
#         ax0.set_xticks(
#             [0, subsets], [f"{worst_score:.3f}", f"{best_score:.3f}"]
#         )
#         ax0.set_xticks([], minor=True)
#         # Disable grids in the two plots. They conflict with the result of
#         # using plotting method 'axvline'
#         ax0.grid(False)
#         ax1.grid(False)

#         # Only adjust the size of the figure here if it is not being plotted
#         # with together with other types of plots in the current figure
#         if self._is_jointplot is False:
#             fig, figsize = self._set_figure_size(
#                 figsize=figsize,
#                 fig=fig,
#                 num_all_features=len(numpy.unique(df["feature_name"])),
#             )

#         return fig, [ax0, ax1]

#     def _apply_styling(
#         self,
#         rcParams: typing.Union[None, dict[str, typing.Any]] = None,
#     ) -> None:
#         # Apply default styling to the generated plots
#         sns.set_theme(style="whitegrid")
#         if rcParams is not None:
#             mpl.rcParams.update(rcParams)
#         else:
#             sns.set_context("paper", rc=DEFAULT_PLOTTING_SETTINGS)


# def plot_roc_curve(
#     df: pandas.DataFrame,
#     figsize: tuple[float, float] = (8, 4),
#     plot_all_curves: bool = True,
#     ax: typing.Union[None, Axes] = None,
#     **kwargs: dict[str, typing.Any],
# ) -> tuple[Figure, Axes]:
#     # Validate given input arguments
#     if ax is None:
#         # No axis was passed as function input argument. Thus create a new
#         # axis object
#         fig, ax = plt.subplots(nrows=1, ncols=1)
#     else:
#         # Get figure from the Axes object so we can subsequently apply
#         # styling to it
#         fig = ax.get_figure()

#     # Set linewidths and alpha values for each of the lines in the ROC AUC
#     # plot
#     line_linewidth = kwargs.get("line_linewidth", 1.25)
#     line_alpha = kwargs.get("line_alpha", 0.30)
#     mean_linewidth = kwargs.get("mean_linewidth", 2.75)
#     mean_alpha = kwargs.get("mean_alpha", 0.75)
#     fill_alpha = kwargs.get("fill_alpha", 0.25)

#     # Make sure valid data pertaining to the 'roc_auc' scoring function is
#     # actually available and set in the 'self.shapfire._history' dataframe
#     try:
#         dict_: dict[str, typing.Any] = df["score"].iat[0]
#         fpr, tpr, roc_auc = dict_["fpr"], dict_["tpr"], dict_["roc_auc"]
#     except KeyError:
#         raise ValueError(
#             "This plotting function can only be called if valid data "
#             + "pertaining to the 'roc_auc' score is availble.."
#         )

#     # Apply default styling
#     _apply_default_styling()

#     # Extract necessary data for plotting
#     tprs = []
#     aucs = []
#     mean_fpr = numpy.linspace(start=0, stop=1, num=100)
#     counter = 1
#     #         for _, row in self.shapfire._history.iterrows():
#     for _, row in df.iterrows():
#         score: dict[str, typing.Any] = row["score"]
#         fpr, tpr, roc_auc = score["fpr"], score["tpr"], score["roc_auc"]
#         # Plot individual ROC lines
#         if plot_all_curves is True:
#             _plot_roc_curve(
#                 ax=ax,
#                 fpr=fpr,
#                 tpr=tpr,
#                 roc_auc=roc_auc,
#                 fold=counter,
#                 line_linewidth=line_linewidth,  # type: ignore
#                 line_alpha=line_alpha,  # type: ignore
#             )
#         counter += 1

#         # Perform one-dimensional linear interpolation for monotonically
#         # increasing sample points
#         interp_tpr = numpy.interp(x=mean_fpr, xp=fpr, fp=tpr)
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)
#         aucs.append(roc_auc)

#     # Plot mean of ROC lines
#     mean_tpr = numpy.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = numpy.std(aucs, ddof=1)
#     ax.plot(
#         mean_fpr,
#         mean_tpr,
#         lw=mean_linewidth,
#         alpha=mean_alpha,
#         color="b",
#         label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
#     )

#     # Fill between ROC lines
#     std_tpr = numpy.std(tprs, axis=0)
#     tprs_upper = numpy.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = numpy.maximum(mean_tpr - std_tpr, 0)
#     ax.fill_between(
#         x=mean_fpr,
#         y1=tprs_lower,
#         y2=tprs_upper,
#         color="gray",
#         alpha=fill_alpha,
#         label=r"$\pm$ 1 std. dev.",
#     )
#     # Random classifier performance line
#     ax.plot([0, 1], [0, 1], color="navy", lw=1.25, linestyle="--")
#     sns.despine(
#         ax=ax,
#         top=True,
#         right=True,
#         left=True,
#         bottom=True,
#         offset=None,
#         trim=True,
#     )

#     # Create a box to fill with legend entries
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 1.0, box.height])
#     ncols = _add_dummy_legend_entries(
#         total_lines=counter, lines_per_inch=4, figsize=figsize, ax=ax
#     )
#     # Put a legend to the right of the current axis
#     legend = ax.legend(
#         loc="center left",
#         bbox_to_anchor=(1, 0.5),
#         ncol=ncols,
#         fancybox=True,
#     )
#     for line in legend.get_lines():
#         line.set_linewidth(2.5)

#     # Make changes related to figure size, title, x-axis labels + ticks,
#     # y-axis labels + and ticks, etc.
#     ax.set_xlim([-0.05, 1.05])
#     ax.set_ylim([-0.05, 1.05])

#     # Position x and y labels manually
#     fig.text(
#         x=0.5,
#         y=0.010,
#         s="False positive rate",
#         ha="center",
#     )
#     fig.text(
#         x=0.0005,
#         y=0.5,
#         s="True positive rate",
#         va="center",
#         rotation="vertical",
#     )
#     fig.subplots_adjust(wspace=0.15, hspace=0.15)
#     ax.set_title("ROC curves")
#     return fig, ax


# def _plot_roc_curve(
#     ax: Axes,
#     fpr: numpy.ndarray,
#     tpr: numpy.ndarray,
#     roc_auc: float,
#     fold: float,
#     line_linewidth: float = 1.25,
#     line_alpha: float = 0.30,
# ) -> Axes:
#     ax.plot(
#         fpr,
#         tpr,
#         lw=line_linewidth,
#         alpha=line_alpha,
#         label=f"Fold {fold}. ROC (AUC = {roc_auc:.2f})",
#     )
#     return ax


# def _add_dummy_legend_entries(
#     total_lines: int,
#     lines_per_inch: float,
#     figsize: tuple[float, float],
#     ax: Axes,
# ) -> int:
#     lines_per_inch = 4
#     total_space = figsize[1] * lines_per_inch
#     ncols = int(numpy.ceil(total_lines / total_space))
#     dummy_lines = int(total_space - (total_lines % total_space))
#     # Add additional dummy legend entries to fill empty space
#     # in the displayed legend
#     for _ in range(dummy_lines):
#         ax.plot([], [], color="black", lw=0, alpha=0, label=" ")
#     return ncols


# def _apply_default_styling(
#     rcParams: typing.Union[None, dict[str, typing.Any]] = None,
# ) -> None:
#     # Apply default styling to the generated plots
#     sns.set_theme(style="whitegrid")
#     if rcParams is not None:
#         mpl.rcParams.update(rcParams)
#     else:
#         sns.set_context("paper", rc=DEFAULT_PLOTTING_SETTINGS)


# def _set_figure_size(
#     figsize: typing.Union[None, tuple[float, float]],
#     fig: Figure,
#     num_all_subsets: typing.Union[None, float, int] = None,
#     num_all_features: typing.Union[None, float, int] = None,
#     default_figsize: typing.Union[None, tuple[float, float]] = None,
# ) -> tuple[Figure, tuple[float, float]]:
#     if figsize is None:
#         if num_all_features is not None:
#             height = 2.5 * numpy.maximum(numpy.rint(num_all_features / 10), 1)
#             if num_all_subsets is not None:
#                 width = 6 * numpy.rint(num_all_subsets / 10)
#             else:
#                 width = 1 * height
#             if default_figsize is not None:
#                 figsize = (
#                     numpy.maximum(width, default_figsize[0]),
#                     numpy.maximum(height, default_figsize[1]),
#                 )
#             else:
#                 figsize = (width, height)
#             fig.set_size_inches(figsize)
#         else:
#             raise ValueError(
#                 "The given input argument 'num_all_features' and "
#                 + "'figsize' is None. If 'figsize' is None, then "
#                 + "'num_all_features' needs to be an integer value "
#                 + "larger than 0."
#             )
#     else:
#         fig.set_size_inches(figsize)
#     return fig, figsize
