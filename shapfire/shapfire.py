import shap
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import squareform
import scipy.stats as stats
from collections import OrderedDict
from operator import itemgetter
from collections import defaultdict
import logging
import typing

from sklearn.base import (
    is_classifier,
    is_regressor,
)
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import auc
import seaborn as sns

import shapfire.utils as utils

# Valid hyperparameter search methods
HYPERPARAMETER_SEARCH_METHODS = [
    RandomizedSearchCV,
    GridSearchCV,
    None,
]

DEFAULT_RANDOM_SEED = 123
LINKAGE_METHODS = [
    "average",
    "centroid",
    "complete",
    "median",
    "single",
    "ward",
]

DEFAULT_SPLITS = 2
"""The default number of folds a dataset should be divided into in a
cross-validation."""

DEFAULT_REPEATS = 1
"""The number of times, in a cross-validation, the division of a dataset into a
certain number of folds should be repeated.
"""

REPLACE = "replace"
"""The default string value used to indicate that NaN or None values should be \
replaced with another given value."""  # pylint: disable=W0105

DROP = "drop"
"""The default string value used to indicate that samples associated with a \
dataset (X) and target variable (y) should be dropped if NaN or None values \
are contained in a sample.
"""  # pylint: disable=W0105

DROP_SAMPLES = "drop_samples"
"""The default string value used to indicate that a sample (row) in a dataset \
(X) should be dropped if it contains NaN or None values.
"""  # pylint: disable=W0105

DROP_FEATURES = "drop_features"
"""The default string value used to indicate that a feature (column) in a \
dataset (X) should be dropped if it contains NaN or None values.
"""  # pylint: disable=W0105

SKIP = "skip"
"""The default string value used to indicate that a value should be skipped \
whenever a NaN or None value is encountered.
"""  # pylint: disable=W0105

DEFAULT_REPLACE_VALUE = 0.0
"""The default value that NaN or None values are replaced with.
"""  # pylint: disable=W0105


def get_kfold_cross_validator(
    estimator_class,
    n_splits,
    n_repeats,
):
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
        raise ValueError


def get_roc_auc_statistics(
    estimator,
    X_test,
    y_test,
):
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


class ShapFire:

    _HISTORY_REQUIRED_FIELDS = [
        "score",
        "feature_importances",
        # Data pertaining to the following fields are not needed anywhere but
        # returned for the sake of convenience in case a user needs the data...
        "shap_values",
    ]

    def __init__(
        self,
        estimator_class,
        scoring,
        estimator_params=None,
        n_splits=2,
        n_repeats=2,
        random_seed=DEFAULT_RANDOM_SEED,
        iterations=None,
    ):
        # Class vars corresponding to input args
        self.estimator_class = estimator_class
        self.scoring = scoring
        self.estimator_params = estimator_params
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.iterations = iterations
        self.random_seed = random_seed

        # Set random seed for reproducibility purposes
        np.random.seed(self.random_seed)

        # Public accessible vars associated with the most important features
        self.selected_features = None
        self._feature_selector = None
        self.cut_value = None
        self.cluster_labels_df = None
        self.feature_ranking_df = None
        self._organized_feature_ranking_df = None

        self._cluster_labels = None
        self._all_feature_names = None


        # Internal vars for easy access to data associated with the importance
        # ranking of features
        self._history = pd.DataFrame()

    def fit(self, X, y):
        # Make sure the column names are strings!
        X.columns = [str(name) for name in X.columns]

        # Determine feature clustering
        self._feature_selector = FeatureSelectionHelper()
        self.cluster_labels_df = self._feature_selector._identify_clusters(X=X)

        if self.iterations is None:
            self.iterations = self._feature_selector.largest_cluster

        # Perform repeated nested Cross-Validation (CV):
        self._outer_cv_loop(X=X, y=y)

        # Calculate normalized SHAP feature importance scores and pick the best
        # feature from each of the previously found clusters
        self._normalized_shap_feature_importance_df = self._calculate_normalized_shap_feature_importance()

        # Select best features
        self.feature_ranking_df = self._pick_top_k_from_clusters(
            df=self._normalized_shap_feature_importance_df,
            top_k=None,
        )

        # Keep clusters in the order they first appear
        # then sort within cluster by importance (descending)
        _feature_ranking_df = self.feature_ranking_df.copy()
        cluster_ordering = (
            _feature_ranking_df.reset_index(level=["cluster"])
            .groupby(by=["cluster"])
            .head(1)["cluster"]
            .values
        )
        _feature_ranking_df = _feature_ranking_df.reset_index(
            level=["cluster"]
        )
        _feature_ranking_df["cluster"] = pd.Categorical(
            _feature_ranking_df["cluster"].values,
            categories=cluster_ordering,
        )
        _feature_ranking_df.sort_values(
            by=["cluster", "normalized_feature_importance"],
            ascending=[True, False],
            inplace=True,
        )
        self._organized_feature_ranking_df = _feature_ranking_df
        self._cluster_labels = self._organized_feature_ranking_df["cluster"].values
        self._all_feature_names = self._organized_feature_ranking_df.index.get_level_values("feature_name").values

        selected_features = (
            self.feature_ranking_df.copy()
            .groupby(level="cluster", sort=False)
            .head(1)
        )

        # Discard features with 0 importance. A selected feature should not have 0 importance!
        selected_features = selected_features[
            selected_features["normalized_feature_importance"] > 0
        ]

        # Further filtering based on a cutoff value
        self.cut_value = self._find_cutoff(df=selected_features)
        selected_features = selected_features[
            selected_features["normalized_feature_importance"] >= self.cut_value
        ]

        selected_feature_names = [v[0] for v in selected_features.index.values]
        self.selected_features = selected_feature_names
        return self

    def _calculate_normalized_shap_feature_importance(self):
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
        folds = self._history.shape[0]
        arr = []
        for i in range(folds):
            df_fold = (
                self._history["feature_importances"]
                .iloc[i]
                .reset_index(drop=True)
            )
            score = self._history["score"].iloc[i][self.scoring]

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
        return pd.DataFrame(data=arr)

    def _pick_top_k_from_clusters(
        self,
        df,
        top_k=None,
    ):
        REQUIRED_FIELDS = [
            "feature_name",
            "cluster",
            "normalized_feature_importance",
        ]
        # Validate input arguments before proceeding
        if not isinstance(df, pd.DataFrame):
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

        if top_k is not None:
            # Return the top k best ranked features from each cluster
            return _df.head(top_k)
        else:
            # Return all features from each cluster
            return _df.head(np.inf)

    def _find_cutoff(self, df: pd.DataFrame, relative_change: float = 0.001) -> float:
        """
        Returns an importance threshold. Keep features that have a larger
        value then the threshold.
        Note that: 'df' has a MultiIndex with level 0 = 'cluster' and a
        column 'normalized_feature_importance' in [0, 1].
        We first take the max per cluster, sort in descending order, then
        pick the largest prefix whose cumulative mass <= total / (1 + relative_change).
        """
        # # Aggregate: largest score per cluster
        s = (
            df.groupby(level=0)["normalized_feature_importance"]
            .max()
            .sort_values(ascending=False)
        )

        # Handle edge cases
        if len(s) == 0:
            return 0.0
        total = float(s.sum())
        if total == 0.0:
            # If we get all zeros, then there is nothing to separate
            # so choose 0 as threshold
            return 0.0

        tau = float(relative_change)
        target = total / (1.0 + tau)

        # Find the largest k with cumsum[k-1] <= target
        cs = s.cumsum().values
        k = np.searchsorted(cs, target, side="right")

        if k <= 0:
            # target is smaller than the top feature; keep at least the top one
            cutoff = float(s.iloc[0])
        else:
            # threshold is the smallest value among the kept features
            cutoff = float(s.iloc[k - 1])

        return cutoff

    def _reorganize_feature_importance_values(
        self,
        df,
    ):
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

    def _get_score(
        self,
        estimator,
        X_test,
        y_test,
    ):
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

                # # Get confusion matrices
                # conf_matrices = get_conf_matrices(
                #     estimator=estimator,
                #     X_test=X_test,
                #     y_test=y_test,
                # )
                # dict_["conf_matrices"] = conf_matrices
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
        X,
        y,
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
                    random_state=self.random_seed, verbosity=-1,
                ).fit(
                    X=_X_train,
                    y=_y_train.ravel(),
                )

                # Retrieve SHAP values on outer loop CV test set using
                # best estimator refitted on inner loop CV training + test set
                shap_values = shap.TreeExplainer(estimator).shap_values(
                    X_test[selected_features]
                )
                values = np.abs(shap_values).mean(axis=0)

                feature_importances = pd.DataFrame(
                    list(zip(selected_features, values)),
                    columns=["feature_name", "feature_importance"],
                )

                feature_importances.sort_values(
                    by=["feature_importance"],
                    ascending=False,
                    inplace=True,
                )
                score = self._get_score(
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

        _history = pd.DataFrame(data=history)
        # If the current ShapFire object already has a 'self._history'
        # defined then reset the dataframe so data does not accumulate
        if self._history is not None:
            self._history = pd.DataFrame()
        self._history = pd.concat(
            [
                self._history.reset_index(drop=True),
                _history.reset_index(drop=True),
            ],
            ignore_index=True,
            join="outer",
            axis=0,
        )

    def plot_importance(
        self,
        plot_type="stripplot",
        groupby="cluster",
        rcParams=None,
        figsize=None,
        fontsize=10,
        with_text=True,
        with_overlay=True,
        ax=None,
    ):
        plotting_interface = ShapFirePlottingInterface(shapfire=self)
        return plotting_interface.plot_importance(
            plot_type=plot_type,
            groupby=groupby,
            rcParams=rcParams,
            figsize=figsize,
            fontsize=fontsize,
            with_text=with_text,
            with_overlay=with_overlay,
            ax=ax,
        )


class FeatureSubsetCollection:
    def __init__(self):
        self.feature_subsets = []
        self._data_dict = {}

    def _add_entries(self, key, data):
        if key in self._data_dict:
            self._data_dict[key] = pd.concat(
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


def cramers_v(
    x,
    y,
    bias_correction,
):
    confusion_matrix = pd.crosstab(index=x, columns=y)
    chi2, _, _, _ = stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    if bias_correction:
        phi2corr = np.maximum(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        if np.minimum((kcorr - 1), (rcorr - 1)) == 0:
            print(
                "Unable to calculate Cramer's V using bias correction. "
                + "Consider using bias_correction=False"
            )
            return np.nan
        else:
            return np.sqrt(phi2corr / np.minimum((kcorr - 1), (rcorr - 1)))
    else:
        if np.minimum(k - 1, r - 1) == 0:
            return np.nan
        else:
            return np.sqrt(phi2 / np.minimum(k - 1, r - 1))


def correlation_ratio(
    categories,
    measurements,
):
    categories = categories.values
    measurements = measurements.values
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(
            n_array,
            np.power(
                np.subtract(
                    y_avg_array,
                    y_total_avg,
                ),
                2,
            ),
        )
    )
    denominator = np.sum(
        np.power(
            np.subtract(
                measurements,
                y_total_avg,
            ),
            2,
        )
    )
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def associations(
    X,
):
    # Extract dataframe column labels
    columns = X.columns

    _X = X.dropna(axis=0, inplace=False)

    # Identify categorical features and columns
    cat_columns = _X.select_dtypes(include=["category"]).columns

    # Create dataframe for storing associations values
    c = pd.DataFrame(index=columns, columns=columns)

    # Find columns consisting of the same value
    single_value_columns_set = set()
    for column in columns:
        if _X[column].unique().size == 1:
            single_value_columns_set.add(column)

    # Compute feature associations
    for i in range(0, len(columns)):
        if columns[i] in single_value_columns_set:
            c.loc[:, columns[i]] = 0.0
            c.loc[columns[i], :] = 0.0
        for j in range(i, len(columns)):
            if columns[j] in single_value_columns_set:
                continue
            elif i == j:
                c.loc[columns[i], columns[j]] = 1.0
            else:
                if columns[i] in cat_columns:
                    if columns[j] in cat_columns:
                        cell = cramers_v(
                            _X[columns[i]],
                            _X[columns[j]],
                            bias_correction=False,
                        )
                        ij, ji = cell, cell
                    else:
                        cell = correlation_ratio(
                            _X[columns[i]],
                            _X[columns[j]],
                        )
                        ij, ji = cell, cell
                else:
                    if columns[j] in cat_columns:
                        cell = correlation_ratio(
                            _X[columns[j]],
                            _X[columns[i]],
                        )
                        ij, ji = cell, cell
                    else:
                        cell, _ = stats.spearmanr(
                            _X[columns[i]],
                            _X[columns[j]],
                        )
                        ij, ji = cell, cell
                c.loc[columns[i], columns[j]] = (
                    ij if not np.isnan(ij) and abs(ij) < np.inf else 0.0
                )
                c.loc[columns[j], columns[i]] = (
                    ji if not np.isnan(ji) and abs(ji) < np.inf else 0.0
                )
    # c.fillna(value=np.nan, inplace=True)
    # return c
    return c.fillna(value=np.nan, inplace=False)


class AutoHierarchicalAssociationClustering:

    def __init__(
        self,
        linkage_methods,
        cluster_distance_threshold=None,
    ):
        """
        Initialize an 'AutoHierarchicalAssociationClustering' object.

        Args:
            linkage_methods: List of possible linkage methods to use in the \
                hierarchical agglomerative clustering of highly \
                associated/correlated features.
            cluster_distance_threshold: The \
                distance threshold to apply when forming flat clusters. \
                Defaults to None.
        """
        # Class variables corresponding to calss input arguments
        self.linkage_methods = linkage_methods
        self.cluster_distance_threshold = cluster_distance_threshold

        # Internal variables for easy access to data associated with the best
        # clustering of features
        self._idx = None
        self._idx_to_cluster_array = None
        self._df = None

        # Publically accessible variables associated with the best clustering
        # of features. These variables wil eventually be set after a call to
        # 'fit()'
        self.clustered_association_matrix = None
        self.linkage_method = None
        self.linkage = None
        self.cophenetic_coeficient = None

    def fit(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError
        # Make sure the matrix is square
        if X.shape[0] != X.shape[1]:
            raise ValueError
        # Make sure input is a similarity matrix consisting of values in the
        # range [-1, 1]. For example correlation is in the range [-1, 1]
        if True in np.unique(X[(X >= -1) & (X <= 1)].isnull()):
            raise ValueError
        # Turn the association matrix X into a dissimilarity matrix
        _X = 1 - np.abs(X)
        # Fill the diagonal elements in the matrix with zeros
        np.fill_diagonal(_X.values, 0)
        # Make sure the matrix is symmetric
        pairwise_distances = squareform(X=_X, checks=False, force="tovector")
        # If no distance threshold is given then use 0.5 as the threshold
        if self.cluster_distance_threshold is None:
            self.cluster_distance_threshold = 0.5
        clustering_info = [{} for _ in range(len(self.linkage_methods))]
        for i in range(0, len(self.linkage_methods)):
            clustering_info[i] = self._perform_feature_clustering(
                X=_X,
                linkage_method=self.linkage_methods[i],
                pairwise_distances=pairwise_distances,
            )

        # Save clustering results in sorted order
        self._df = pd.DataFrame(data=clustering_info).sort_values(
            by=["cophenetic_coefficient"],
            ascending=False,
        )
        # Set the best parameter values that have been found
        row = self._df.iloc[0]
        self._idx = row["idx"]
        self._idx_to_cluster_array = row["idx_to_cluster_array"]
        self.linkage_method = row["linkage_method"]
        self.cophenetic_coefficient = row["cophenetic_coefficient"]

        # Continue if the best parameter values have been set correctly
        if self._idx is not None:
            self.clustered_association_matrix = X.iloc[self._idx, :].T.iloc[
                self._idx, :
            ]
            # return self
            return self._idx_to_cluster_array
        else:
            raise ValueError(
                "Internal error. The indexing array '._idx' is None. "
            )

    def _perform_feature_clustering(
        self,
        X,
        linkage_method,
        pairwise_distances,
    ):
        """
        Given the necessary data, perform hierachical agglomerative clustering
        and evaluate the quality of the obtained clustering.

        Args:
            X: The dataset whose features are to be clustered.
            linkage_method: The linkage method to use when applying \
                hierachical agglomerative clustering to group highly \
                associated/correlated features.
            pairwise_distances: The pairwise distances between features, of a \
                dataset, representing the dissimilarity between features.

        Returns:
            Data pertaining to the obtained clutering of features along with \
            different statistics that can be used to evaluate the quality of \
            the obtained clustering.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "The given input arugment 'X' should be of type "
                + f"'DataFrame' but is intead of type '{type(X)}'."
            )
        # A linkage method is used to compute the distance between two clusters
        linkage = hac.linkage(y=pairwise_distances, method=linkage_method)
        idx_to_cluster_array = hac.fcluster(
            Z=linkage,
            t=self.cluster_distance_threshold,
            # criterion="distance" --> Forms flat clusters so that the original
            # observations in each flat cluster have no greater a cophenetic
            # distance than t = self.cluster_distance_threshold.
            criterion="distance",
        )
        idx = np.argsort(idx_to_cluster_array)

        # Compute the cophenetic correlation coefficient
        cophenetic_coef, _ = hac.cophenet(Z=linkage, Y=pairwise_distances)
        cluster_labels = np.unique(idx_to_cluster_array)

        # Return data associated with the obtained clustering so we subsequently
        # can determine the best approach
        return {
            "linkage_method": linkage_method,
            "cophenetic_coefficient": cophenetic_coef,
            "idx_to_cluster_array": idx_to_cluster_array,
            "idx": idx,
            "linkage_object": linkage,
        }


def _identify_colinear_features(
    # NOTE: Internal method. Assume 'df' is passed as a pd dataframe
    df,
    linkage_methods=LINKAGE_METHODS,
):
    # Determine the pairwise strength of association/correlation between features
    feature_associations = associations(X=df)

    # Cluster collinear/multicollinear features
    _idx_to_cluster_array = AutoHierarchicalAssociationClustering(
        linkage_methods=linkage_methods
    ).fit(feature_associations)

    # Organize information in a dictionary and then a dataframe
    # Use defaultdict to automatically create empty lists as values for missing keys
    dict_cluster_labels = defaultdict(list)

    # # Extract a list of feature names
    feature_names = df.columns.to_list()

    # Extract all cluster labels
    if _idx_to_cluster_array is not None:
        # Populate each of the lists associated with a cluster with feature names
        for i in range(len(_idx_to_cluster_array)):
            cluster_label = _idx_to_cluster_array[i]
            dict_cluster_labels[cluster_label].append(str(feature_names[i]))

        # Convert defaultdict back to a regular dict if necessary
        dict_cluster_labels = dict(dict_cluster_labels)

        lst = []
        for label in dict_cluster_labels:
            for feature_name in dict_cluster_labels[label]:
                d = {"cluster_label": label, "feature_name": feature_name}
                lst.append(d)
        df_cluster_labels = pd.DataFrame(data=lst)
        return df_cluster_labels
    else:
        raise ValueError


class FeatureSelectionHelper:

    def __init__(
        self,
    ):
        self._cluster_labels_df = None

    @property
    def nclusters(self):
        if self._cluster_labels_df is not None:
            return np.unique(
                self._cluster_labels_df["cluster_label"].values
            ).shape[0]
        else:
            raise ValueError

    @property
    def largest_cluster(self):
        cluster_size_max = 0
        if self._cluster_labels_df is not None:
            for _, df in self._cluster_labels_df.groupby("cluster_label"):
                cluster_size = df.shape[0]
                if cluster_size > cluster_size_max:
                    cluster_size_max = cluster_size
            return cluster_size_max
        else:
            raise ValueError

    @property
    def feature_clusters(self):
        if self._cluster_labels_df is not None:
            feature_clusters = []
            for _, df in self._cluster_labels_df.groupby("cluster_label"):
                feature_clusters.append(df["feature_name"].to_list())
            return feature_clusters
        else:
            raise ValueError

    def _identify_clusters(
        self,
        X,
    ):
        _X = X.dropna(axis=0)
        self._cluster_labels_df = _identify_colinear_features(df=_X)
        return self._cluster_labels_df


class Cluster:
    def __init__(self, feature_names):
        _feature_names = {label: 0 for label in feature_names}
        self.feature_names = OrderedDict(
            sorted(_feature_names.items(), key=itemgetter(1))
        )

    def get_feature(self):
        return self._first(self.feature_names)

    def update_counter(self, feature_name):
        self.feature_names[feature_name] += 1
        # Update the odered dictionary with new counts
        self.feature_names = OrderedDict(
            sorted(self.feature_names.items(), key=itemgetter(1))
        )

    def _first(self, collection):
        """
        Return the first element from an ordered collection or an arbitrary
        element from an unordered collection.

        Raise StopIteration if the collection is empty.
        """
        return next(iter(collection))


class ClusterSampler:
    def __init__(self, feature_clusters):
        self.clusters = {}
        counter = 0
        for feature_names in feature_clusters:
            self.clusters[f"cluster{counter}"] = Cluster(
                feature_names=feature_names
            )
            counter += 1

    def sample_feature_subset(self):
        feature_subset = []
        for cluster_label in self.clusters:
            feature_name = self.clusters[cluster_label].get_feature()
            feature_subset.append(feature_name)
            self.clusters[cluster_label].update_counter(feature_name)
        return feature_subset


# Define the main color palette to use for plots and other illustations
MAIN_COLOR_PALETTE = {
    "background": "#feffff",
    "selected": "#25c5da",
    "rejected": "#010105",
    "secondary": "#c53e6e",
    "tertiary": "#3e77bf",
    "overlay": "#9c9d9d",
    "grid": "#d2d2d2",
}

# Define the main matplotlib and seaborn plotting settings
DEFAULT_PLOTTING_SETTINGS = {
    "axes.facecolor": MAIN_COLOR_PALETTE["background"],
    "patch.edgecolor": MAIN_COLOR_PALETTE["background"],
    "figure.facecolor": MAIN_COLOR_PALETTE["background"],
    "axes.edgecolor": MAIN_COLOR_PALETTE["background"],
    "savefig.edgecolor": MAIN_COLOR_PALETTE["background"],
    "savefig.facecolor": MAIN_COLOR_PALETTE["background"],
    "grid.color": MAIN_COLOR_PALETTE["grid"],
    "lines.linewidth": 1.30,
}


class ShapFirePlottingInterface:
    def __init__(self, shapfire):  # type: ignore
        # Class vars corresponding to input args
        self.shapfire = shapfire  # noqa

        # Internal vars for easy access to data associated with the importance
        # ranking of features
        self._data = None
        self._is_jointplot = False
        self._groupby = None

    def _organize_data_for_plotting(
        self,
        feature_ranking_df,
        df,
        groupby,
    ):
        """
        Organize and structure the results obtained by applying ShapFire such
        that the results can easily be plotted and displayed in a figure.

        Args:
            feature_ranking_df: _description_
            df: A dataframe containing all the necessary data for visualizing \
                the importance ranking of features.
            groupby: A string value indicating how the feature importance \
                ranking should be displayed in a figure. If the option \
                'cluster' is chosen, then the features are grouped and shown \
                in the figure based on their assigned cluster and according to \
                the importance rank of the best feautre in the cluster. If \
                'feature' is chosen, then the features are shown in the figure \
                purely according to their global rank without any \
                consideration to what cluster each features are a part of.

        Raises:
            TypeError: If the input argument 'groupby' is not a string.
            ValueError: If the input argument 'groupby' is not a valid option.

        Returns:
            Organized and structured data of ShapFire results that can be \
            passed on to appropriate plotting methods.
        """
        if not isinstance(groupby, str):
            raise TypeError(
                "Function argument 'groupby' should be of type 'str' but "
                + f"argument of type {type(groupby)} was given."
            )
        else:
            names = np.unique(df["feature_name"]).tolist()
            template_dict = {n: np.nan for n in names}
            data = []
            if groupby.strip().lower() == "feature":
                indexing = (
                    df[["feature_name", "normalized_feature_importance"]]
                    .groupby("feature_name")
                    .agg("median")
                    .sort_values(
                        by=["normalized_feature_importance"],
                        ascending=True,
                    )
                    .index
                )
                for _name, _df in df[
                    ["feature_name", "normalized_feature_importance"]
                ].groupby("feature_name"):
                    for _, _row in _df.iterrows():
                        dict_ = template_dict.copy()
                        dict_[_name] = _row["normalized_feature_importance"]
                        data.append(dict_)
                # Return tuple:
                # - Data
                # - Vertical ordering by feature name according to feature
                #   importance rank
                new_df = pd.DataFrame(data=data).reindex(indexing, axis=1)
                return new_df, list(reversed(new_df.columns.values))
            elif groupby.strip().lower() == "cluster":
                indexing = feature_ranking_df.index.values
                for _name, _df in df[
                    ["feature_name", "normalized_feature_importance"]
                ].groupby("feature_name"):
                    for _, _row in _df.iterrows():
                        dict_ = template_dict.copy()
                        dict_[_name] = _row["normalized_feature_importance"]
                        data.append(dict_)
                # Return tuple:
                # - Data
                # - Vertical ordering by feature name according to feature
                #   importance rank and cluster label
                new_df = pd.DataFrame(data=data).reindex(indexing, axis=1)
                return new_df, list(new_df.columns.values)
            else:
                raise ValueError(
                    "The given input argument 'groupby' should have value "
                    + f"'feature' or 'cluster' but a value '{groupby}' was "
                    + "given."
                )

    def _prepare_data(self):
        plotting_df, feature_ordering = self._organize_data_for_plotting(
            df=self.shapfire._normalized_shap_feature_importance_df,
            feature_ranking_df=self.shapfire._organized_feature_ranking_df,
            groupby="cluster",
        )

        # Set colors for each selected/rejected feature
        feature_colors = {}
        for feature_name in self.shapfire._all_feature_names:
            if feature_name in self.shapfire.selected_features:
                feature_colors[feature_name] = MAIN_COLOR_PALETTE["selected"]
            else:
                feature_colors[feature_name] = MAIN_COLOR_PALETTE["rejected"]

        return {
            # Main importance plot fields...
            "df": self.shapfire._normalized_shap_feature_importance_df,
            "feature_ordering": feature_ordering,
            "feature_colors": feature_colors,
            "selected_features": self.shapfire.selected_features,
            # "all_features": all_features,
            # "all_features": None,
            "all_feature_names": self.shapfire._all_feature_names,
            "cluster_labels": self.shapfire._cluster_labels,
            "plotting_df": plotting_df,
        }

    def _add_cluster_overlays(
        self,
        ax,
        cluster_labels,
        fontsize=10,
        with_text=True,
        with_overlay=True,
        x_offset=0,
    ):
        # y-offset. Move text slightly down
        text_placement_offset = 0.00
        current_cluster_label = cluster_labels[0]
        lower_value = -0.5
        upper_value = 0.5
        last_index = len(cluster_labels[1:]) + 1

        # Alpha values associated with the two alternating overlays
        alphas = [0.05, 0.25]

        # Values pertaining to first cluster overlay
        counter0 = 1
        counter1 = 0
        alpha = alphas[(counter1 + 1) % 2]

        # Add overlays by looping over cluster labels associated
        # with each feature present in the input dataset
        for i in range(1, last_index):
            if current_cluster_label != cluster_labels[i]:
                if with_overlay is True:
                    ax.axhline(upper_value, color="black", alpha=0.10)
                    ax.axhspan(
                        lower_value,
                        upper_value,
                        facecolor=MAIN_COLOR_PALETTE["overlay"],
                        alpha=alpha,
                    )
                if with_text is True:
                    ax.text(
                        x=x_offset,
                        # Text placement
                        y=lower_value
                        + (upper_value - lower_value) / 2
                        + text_placement_offset,
                        s=f"Cluster {current_cluster_label}",
                        fontsize=fontsize,
                        verticalalignment="center",
                    )
                counter0 = 1
                counter1 += 1
                alpha = alphas[(counter1 + 1) % 2]
                lower_value = upper_value
                upper_value += 1.00
            else:
                counter0 += 1
                upper_value += 1.00
            current_cluster_label = cluster_labels[i]
        if with_overlay is True:
            ax.axhspan(
                lower_value,
                upper_value,
                facecolor=MAIN_COLOR_PALETTE["overlay"],
                alpha=alpha,
            )
        if with_text is True:
            ax.text(
                x=x_offset,
                # Text placement
                y=lower_value
                + (upper_value - lower_value) / 2
                + text_placement_offset,
                s=f"Cluster {current_cluster_label}",
                fontsize=fontsize,
                verticalalignment="center",
            )

    def _add_feature_overlays(
        self,
        ax,
        cluster_labels,
    ):
        cluster_labels[0]
        lower_value = -0.5
        upper_value = 0.5
        last_index = len(cluster_labels[1:]) + 1

        # Opacity values associated with the two alternating overlays
        alphas = [0.05, 0.25]

        # Values pertaining to first cluster overlay
        counter1 = 0
        alpha = alphas[(counter1 + 1) % 2]

        # Add overlays
        for _ in range(1, last_index):
            ax.axhline(upper_value, color="black", alpha=0.10)
            ax.axhspan(
                lower_value,
                upper_value,
                facecolor=MAIN_COLOR_PALETTE["overlay"],
                alpha=alpha,
            )
            counter1 += 1
            alpha = alphas[(counter1 + 1) % 2]
            lower_value = upper_value
            upper_value += 1.00
        ax.axhspan(
            lower_value,
            upper_value,
            facecolor=MAIN_COLOR_PALETTE["overlay"],
            alpha=alpha,
        )

    def ceil5(self, x):
        """
        Given an input value round the value to closest and largest multiple of
        5.

        Args:
            x: A value that is to be rounded.

        Returns:
            The input value rounded to the closest and largest multiple
            of 5.
        """
        base = 5
        return int(base * np.ceil(x / base))

    def plot_importance(
        self,
        plot_type="stripplot",
        groupby="cluster",
        rcParams=None,
        figsize=None,
        fontsize=10,
        with_text=True,
        with_overlay=True,
        ax=None,
    ):
        # Define the default plotting options
        PLOT_IMPORTANCE_OPTIONS = {
            # Do not allow violinplot. The elements will be squished too
            # much and result in an awful representation of the data
            "stripplot": {
                "func": sns.stripplot,
                "xargs": {"dodge": True, "alpha": 0.66, "ax": ax},
            },
            "swarmplot": {
                "func": sns.swarmplot,
                "xargs": {},
            },
            "boxplot": {
                "func": sns.boxplot,
                "xargs": {
                    "medianprops": {
                        "color": "white",
                        "linewidth": 1.25,
                    },
                    "boxprops": {
                        "linewidth": 0.5,
                    },
                    "whiskerprops": {
                        "linewidth": 1.5,
                    },
                    "capprops": {
                        "linewidth": 1.5,
                    },
                },
            },
        }

        # Validate given input arguments
        if not isinstance(plot_type, str):
            raise TypeError(
                "The given input argument 'plot_type' should be of type "
                + f"'str' but argument of type '{type(plot_type)}' was given."
            )
        else:
            _PLOT_OPTIONS = list(PLOT_IMPORTANCE_OPTIONS.keys())
            if not plot_type.strip().lower() in _PLOT_OPTIONS:
                raise ValueError(
                    "The given input argument 'plot_type' should be one "
                    + f"of the following options: {', '.join(_PLOT_OPTIONS)} "
                    + f" but an argument '{plot_type}' was given."
                )

        if not isinstance(groupby, str):
            raise TypeError(
                "The given input argument 'groupby' should be of type "
                + f"'str' but an argument of type '{type(groupby)}' was given."
            )
        else:
            GROUBPBY_OPTIONS = ["feature", "cluster"]
            if not groupby.strip().lower() in GROUBPBY_OPTIONS:
                raise ValueError(
                    "The given input argument 'plot_type' should be one "
                    + f"of the following options: {', '.join(_PLOT_OPTIONS)} "
                    + f" but an argument '{plot_type}' was given."
                )
        if ax is None:
            # No axis was passed as function input argument. Thus create a new
            # axis object
            fig, ax = plt.subplots(nrows=1, ncols=1)
        else:
            # Get figure from the Axes object so we can subsequently apply
            # styling to it
            fig = ax.get_figure()

        # Apply styling to the plot elements
        self._apply_styling(rcParams)

        # Prepare the appropriate data for plotting
        _groupby = groupby.strip().lower()
        if self._data is None or _groupby != self._groupby:
            # self._data = self._prepare_data(groupby=_groupby)
            self._data = self._prepare_data()
            self._groupby = _groupby

        # Unpack all necessary data for plotting
        df = self._data["df"]
        feature_ordering = self._data["feature_ordering"]
        feature_colors = self._data["feature_colors"]
        cluster_labels = self._data["cluster_labels"]

        # Determine the searborn function to use for plotting and set function
        # arguments that should be passed to the plotting function
        args = {
            "x": "normalized_feature_importance",
            "y": "feature_name",
            "data": df,
        }

        plotting_function = PLOT_IMPORTANCE_OPTIONS[plot_type]["func"]
        args.update(PLOT_IMPORTANCE_OPTIONS[plot_type]["xargs"])
        ax = plotting_function(
            order=feature_ordering,
            palette=list(feature_colors.values()),
            **args,
        )
        sns.despine(
            ax=ax,
            top=True,
            right=True,
            left=True,
            bottom=True,
            offset=None,
            trim=False,
        )

        x_max = df["normalized_feature_importance"].max().max()
        df["normalized_feature_importance"].min().min()
        ax.set_xlim([0.00 - 0.025, x_max + 0.025])

        # Add additional plot overlays depending on how features should be
        # grouped and displayed in the plot
        if groupby.strip().lower() == "cluster":
            if with_text is True or with_overlay is True:
                # Add two alternating gray-scale colors for grouping features
                # based on the cluster they each belong to. Also, add text
                # information about the cluster each feature belongs to
                self._add_cluster_overlays(
                    ax=ax,
                    cluster_labels=cluster_labels,
                    with_text=with_text,
                    with_overlay=with_overlay,
                    x_offset=x_max * 1.10,
                )
        # elif groupby.strip().lower() == "feature":
        #     if with_overlay is True:
        #         # Add two alternating gray-scale colors for better seperation
        #         # of the plotted data. By default do not add text information
        #         # about the clusters each feature belong to. For this purpose,
        #         # the groupby = "cluster" should be chosen
        #         self._add_feature_overlays(ax=ax, cluster_labels=cluster_labels)
        else:
            raise ValueError(
                "The given input argument 'groupby' should have value "
                + f"'feature' or 'cluster' but value '{groupby}' was given."
            )

        # Add a legend to the figure indicating which feautres have been
        # selected and which have been rejected
        custom_lines = [
            Line2D(
                [0],
                [0],
                color=MAIN_COLOR_PALETTE["selected"],
                lw=4.5,
            ),
            Line2D(
                [0],
                [0],
                color=MAIN_COLOR_PALETTE["rejected"],
                lw=4.5,
            ),
        ]
        ax.legend(
            custom_lines,
            ["Selected", "Rejected"],
            loc="lower right",
            fontsize=fontsize + 1,
        )

        # Make changes related to figure size, title, x-axis labels + ticks,
        # y-axis labels + and ticks, etc.
        ax.set_title(
            "ShapFire importance ranking and selected features",
            fontsize=fontsize + 1,
            pad=20,
        )
        ax.set_xlabel(
            "Normalized SHAP feature importance", fontsize=fontsize + 1
        )
        # Only display y-axis label if it is plotted alone
        if self._is_jointplot is False:
            ax.set_ylabel("Feature name", fontsize=fontsize + 1)
        else:
            ax.set_ylabel(None)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=fontsize)
        ax.set_zorder(1)
        plt.style.use('ggplot')
        plt.margins(x=0,y=0)

        # ax.set_ylim([0, len(np.unique(ndf["feature_name"].values)) + 1])

        fig = ax.get_figure()
        fig.set_size_inches(12, 16)
        plt.tight_layout(pad=0)
        return fig, ax

    def _apply_styling(
        self,
        rcParams=None,
    ) -> None:
        # Apply default styling to the generated plots
        sns.set_theme(style="whitegrid")
        if rcParams is not None:
            mpl.rcParams.update(rcParams)
        else:
            sns.set_context("paper", rc=DEFAULT_PLOTTING_SETTINGS)


# def plot_roc_curve(
#     df,
#     figsize=(8, 4),
#     plot_all_curves=True,
#     ax=None,
#     **kwargs,
# ):
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
#         dict = df["score"].iat[0]
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
#     mean_fpr = np.linspace(start=0, stop=1, num=100)
#     counter = 1
#     #         for _, row in self.shapfire._history.iterrows():
#     for _, row in df.iterrows():
#         score = row["score"]
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
#         interp_tpr = np.interp(x=mean_fpr, xp=fpr, fp=tpr)
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)
#         aucs.append(roc_auc)

#     # Plot mean of ROC lines
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = np.std(aucs, ddof=1)
#     ax.plot(
#         mean_fpr,
#         mean_tpr,
#         lw=mean_linewidth,
#         alpha=mean_alpha,
#         color="b",
#         label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
#     )

#     # Fill between ROC lines
#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
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
#     ax,
#     fpr,
#     tpr,
#     roc_auc,
#     fold,
#     line_linewidth=1.25,
#     line_alpha=0.30,
# ):
#     ax.plot(
#         fpr,
#         tpr,
#         lw=line_linewidth,
#         alpha=line_alpha,
#         label=f"Fold {fold}. ROC (AUC = {roc_auc:.2f})",
#     )
#     return ax


def plot_roc_curve(
    df,
    figsize = (8, 4),
    plot_all_curves = True,
    ax = None,
    **kwargs,
):
    # Validate given input arguments
    if ax is None:
        # No axis was passed as function input argument. Thus create a new
        # axis object
        fig, ax = plt.subplots(nrows=1, ncols=1)
    else:
        # Get figure from the Axes object so we can subsequently apply
        # styling to it
        fig = ax.get_figure()

    # Set linewidths and alpha values for each of the lines in the ROC AUC
    # plot
    line_linewidth = kwargs.get("line_linewidth", 1.25)
    line_alpha = kwargs.get("line_alpha", 0.30)
    mean_linewidth = kwargs.get("mean_linewidth", 2.75)
    mean_alpha = kwargs.get("mean_alpha", 0.75)
    fill_alpha = kwargs.get("fill_alpha", 0.25)

    # Make sure valid data pertaining to the 'roc_auc' scoring function is
    # actually available and set in the 'self.shapfire._history' dataframe
    try:
        dict_: dict[str, typing.Any] = df["score"].iat[0]
        fpr, tpr, roc_auc = dict_["fpr"], dict_["tpr"], dict_["roc_auc"]
    except KeyError:
        raise ValueError(
            "This plotting function can only be called if valid data "
            + "pertaining to the 'roc_auc' score is availble.."
        )

    # Apply default styling
    _apply_default_styling()

    # Extract necessary data for plotting
    tprs = []
    aucs = []
    mean_fpr = np.linspace(start=0, stop=1, num=100)
    counter = 1
    #         for _, row in self.shapfire._history.iterrows():
    for _, row in df.iterrows():
        score: dict[str, typing.Any] = row["score"]
        fpr, tpr, roc_auc = score["fpr"], score["tpr"], score["roc_auc"]
        # Plot individual ROC lines
        if plot_all_curves is True:
            _plot_roc_curve(
                ax=ax,
                fpr=fpr,
                tpr=tpr,
                roc_auc=roc_auc,
                fold=counter,
                line_linewidth=line_linewidth,  # type: ignore
                line_alpha=line_alpha,  # type: ignore
            )
        counter += 1

        # Perform one-dimensional linear interpolation for monotonically
        # increasing sample points
        interp_tpr = np.interp(x=mean_fpr, xp=fpr, fp=tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

    # Plot mean of ROC lines
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs, ddof=1)
    ax.plot(
        mean_fpr,
        mean_tpr,
        lw=mean_linewidth,
        alpha=mean_alpha,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    )

    # Fill between ROC lines
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        x=mean_fpr,
        y1=tprs_lower,
        y2=tprs_upper,
        color="gray",
        alpha=fill_alpha,
        label=r"$\pm$ 1 std. dev.",
    )
    # Random classifier performance line
    ax.plot([0, 1], [0, 1], color="navy", lw=1.25, linestyle="--")
    sns.despine(
        ax=ax,
        top=True,
        right=True,
        left=True,
        bottom=True,
        offset=None,
        trim=True,
    )

    # Create a box to fill with legend entries
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1.0, box.height])
    ncols = _add_dummy_legend_entries(
        total_lines=counter, lines_per_inch=4, figsize=figsize, ax=ax
    )
    # Put a legend to the right of the current axis
    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=ncols,
        fancybox=True,
    )
    for line in legend.get_lines():
        line.set_linewidth(2.5)

    # Make changes related to figure size, title, x-axis labels + ticks,
    # y-axis labels + and ticks, etc.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    # Position x and y labels manually
    fig.text(
        x=0.5,
        y=0.010,
        s="False positive rate",
        ha="center",
    )
    fig.text(
        x=0.0005,
        y=0.5,
        s="True positive rate",
        va="center",
        rotation="vertical",
    )
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    ax.set_title("ROC curves")
    return fig, ax


def _plot_roc_curve(
    ax,
    fpr,
    tpr,
    roc_auc,
    fold,
    line_linewidth = 1.25,
    line_alpha = 0.30,
):
    ax.plot(
        fpr,
        tpr,
        lw=line_linewidth,
        alpha=line_alpha,
        label=f"Fold {fold}. ROC (AUC = {roc_auc:.2f})",
    )
    return ax



def _add_dummy_legend_entries(
    total_lines,
    lines_per_inch,
    figsize,
    ax,
):
    lines_per_inch = 4
    total_space = figsize[1] * lines_per_inch
    ncols = int(np.ceil(total_lines / total_space))
    dummy_lines = int(total_space - (total_lines % total_space))
    # Add additional dummy legend entries to fill empty space
    # in the displayed legend
    for _ in range(dummy_lines):
        ax.plot([], [], color="black", lw=0, alpha=0, label=" ")
    return ncols


def _apply_default_styling(
    rcParams=None,
):
    # Apply default styling to the generated plots
    sns.set_theme(style="whitegrid")
    if rcParams is not None:
        mpl.rcParams.update(rcParams)
    else:
        sns.set_context("paper", rc=DEFAULT_PLOTTING_SETTINGS)

class RefitHelper:
    def __init__(
        self,
        feature_names,
        estimator_class,
        scoring,
        estimator_params,
        n_splits=DEFAULT_SPLITS,
        n_repeats=DEFAULT_REPEATS,
        random_seed=utils.DEFAULT_RANDOM_SEED,
    ):
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
        # self._check_vars()

        # Set random seed for reproducibility purposes
        np.random.seed(self.random_seed)

        # Public accessible vars associated with the most important features
        # These vars wil eventually be set after a call to 'fit()'
        self.history = pd.DataFrame()

    def fit(self, X, y):
        history = []
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

            score = self._get_score(
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

        _history = pd.DataFrame(data=history)
        # If the current ShapFire object already has a 'self.history'
        # defined then reset the dataframe so data does not accumulate
        if self.history is not None:
            self.history = pd.DataFrame()
        self.history = pd.concat(
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
        estimator,
        X_test,
        y_test,
    ):
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

                # Get confusion matrices
                conf_matrices = get_conf_matrices(
                    estimator=estimator,
                    X_test=X_test,
                    y_test=y_test,
                )
                dict_["conf_matrices"] = conf_matrices

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

class HyperparameterSearchHelper:
    """
    A ShapFire helper class for performing cross-validation and hyperparameter
    tuning.

    Args:
        BaseEstimator: A scikit-learn estimator class used for API \
            compatibility purposes.
    """

    def __init__(
        self,
        cv,
        estimator_class,
        estimator_params,
        scoring,
        hyperparameter_search=None,
        n_jobs=None,
        random_seed=utils.DEFAULT_RANDOM_SEED,
    ):
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

        # Publically accessible variables associated with the model that
        # obtained the best performance score. These variables wil eventually be
        # set after a call to 'fit()'
        self.best_score = None
        self.best_params = None

    def fit(
        self,
        X,
        y,
    ):
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
            means = np.mean(cv_scores)
            stds = np.std(cv_scores, ddof=1)
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


def get_conf_matrices(estimator, X_test, y_test, n_points=20):
    # TP = confusion[1, 1] is true positives
    # TN = confusion[0, 0] is true negatives
    # FP = confusion[0, 1] is false positives
    # FN = confusion[1, 0] is false negatives
    increment = 1 / n_points
    thresholds = [i * increment for i in range(n_points + 1)]
    results = []
    for probability in thresholds:
        y_pred = (estimator.predict_proba(X_test)[:, 1] >= probability).astype(
            bool
        )
        conf_matrix = confusion_matrix(y_test.astype(bool), y_pred)
        total = np.sum(np.sum(conf_matrix))
        accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / total
        specificity = conf_matrix[0, 0] / (
            conf_matrix[0, 0] + conf_matrix[0, 1]
        )
        sensitivity = conf_matrix[1, 1] / (
            conf_matrix[1, 0] + conf_matrix[1, 1]
        )
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


def _check_hyperparameter_search_params(
    hyperparameter_search,
):
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


def hyperparameter_search_helper(
    X,
    y,
    feature_names,
    estimator_class,
    estimator_params,
    scoring,
    n_splits,
    n_repeats,
    hyperparameter_search=None,
    n_jobs=None,
    random_seed=utils.DEFAULT_RANDOM_SEED,
):
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
