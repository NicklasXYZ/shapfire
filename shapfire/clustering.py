"""This file contains methods can be applied for the purpose of clustering
highly associated/correlated features."""

import faulthandler

faulthandler.enable()

import logging
import typing
import warnings

from collections import OrderedDict
from operator import itemgetter

import numpy
import pandas
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator

# from tqdm import tqdm

import shapfire.utils as utils

LINKAGE_METHODS: list[str] = [
    "average",
    "centroid",
    "complete",
    "median",
    "single",
    "ward",
]


CLUSTERING_CRITERIA: dict[str, typing.Any] = {
    # Name : Sort score in ascending order
    # - 'cophenetic_coefficient': Larger is better
    "cophenetic_coefficient": False,
}


class AutoHierarchicalAssociationClustering(BaseEstimator):
    """An automated approach to feature clustering based on bi-variate
    association/correlation measures and hierarchical agglomerative
    clustering."""

    def __init__(
        self,
        linkage_methods: list[str] = LINKAGE_METHODS,
        cluster_distance_threshold: typing.Union[None, float] = None,
        refit_criteria: str = "cophenetic_coefficient",
    ) -> None:
        """
        Initialize an 'AutoHierarchicalAssociationClustering' object.

        Args:
            linkage_methods: List of possible linkage methods to use in the \
                hierarchical agglomerative clustering of highly \
                associated/correlated features.
            cluster_distance_threshold: The \
                distance threshold to apply when forming flat clusters. \
                Defaults to None.
            refit_criteria: The criterion for picking the best hierarchical \
                agglomerative clustering method. Defaults to \
                "cophenetic_coefficient".
        """
        # Class variables corresponding to calss input arguments
        self.linkage_methods = linkage_methods
        self.cluster_distance_threshold = cluster_distance_threshold
        self.refit_criteria = refit_criteria
        # Check that the given input is valid
        self._check_vars()

        # Internal variables for easy access to data associated with the best
        # clustering of features
        self._idx: typing.Union[None, numpy.ndarray] = None
        self._idx_to_cluster_array: typing.Union[None, numpy.ndarray] = None
        self._df: typing.Union[None, pandas.DataFrame] = None

        # Private variable for tracking the progress of the feature clustering
        self._progress_bar = None

        # Publically accessible variables associated with the best clustering
        # of features. These variables wil eventually be set after a call to
        # 'fit()'
        self.clustered_association_matrix: typing.Union[None, pandas.DataFrame] = None
        self.linkage_method: typing.Union[None, str] = None
        self.linkage: typing.Union[None, str] = None
        self.cophenetic_coeficient: typing.Union[None, float] = None

    def fit(
        self,
        X: typing.Union[numpy.ndarray, pandas.DataFrame],
        y: None = None,
    ) -> "AutoHierarchicalAssociationClustering":
        """
        Perform hierachical agglomerative clustering.

        Args:
            X: Input dataset consisting of several features that are possibly \
                highly associated/correlated and that are to be clustered.
            y: Not used, present for API consistency by convention. Defaults \
                to None.

        Raises:
            ValueError: If the given input argument 'X' does not have the same \
                number of rows and columns. 'X' needs to be symmetric.
            ValueError: If the given input argument 'X' is not a similarity \
                matrix consisting of values in the range [-1, 1].
            ValueError: If the internally set indexing array '._idx' is None.

        Returns:
            An object containing all data associated with the best possible
            clustering of features.
        """
        logging.info("Applying Hierarchical Agglomerative Clustering...")
        if isinstance(X, numpy.ndarray):
            logging.info("Converting input ndarray 'X' to a DataFrame.")
            X = pandas.DataFrame(data=X)
        elif isinstance(X, pandas.DataFrame):
            pass
        else:
            raise ValueError(
                "The given input argument 'X' is not of type "
                + "'ndarray' or 'DataFrame'. 'X' is instead "
                + f"of type {type(X)}."
            )
        if y is not None:
            msg = "Input argument 'y' has been provided but it will " + "not be used!"
            logging.warning(msg)
            warnings.warn(message=msg)

        # Compute feature associations/correlations
        feature_associations = utils.associations(X=X)

        # # Make sure the matrix is square
        if feature_associations.shape[0] != feature_associations.shape[1]:
            raise ValueError(
                "The given input argumnet 'X' does not have the same "
                + "number of rows and columns. 'X' needs to be symmetric."
            )
        # Make sure input is a similarity matrix consisting of values in the
        # range [-1, 1]. For example correlation is in the range [-1, 1]
        if True in numpy.unique(
            feature_associations[
                (feature_associations >= -1) & (feature_associations <= 1)
            ].isnull()
        ):
            raise ValueError(
                "The given input argument 'X' is not a similarity matrix "
                + "consisting of values in therange [-1, 1]."
            )

        # Turn the association matrix X into a dissimilarity matrix
        _X = 1 - numpy.abs(feature_associations)
        # Fill the diagonal elements in the matrix with zeros
        numpy.fill_diagonal(_X.values, 0)
        # Make sure the matrix is symmetric
        pairwise_distances = squareform(X=_X, checks=False, force="tovector")

        # Create progress bar which will be updated continuously
        # to track the progress of the feature clustering
        # self._progress_bar = tqdm(
        #     total=len(self.linkage_methods),
        #     unit_scale=True,
        #     ascii=" >=",
        #     bar_format="{desc:<20}{percentage:3.0f}%|{bar:25}{r_bar}",
        #     desc="Clustering progress ",
        # )

        # If no distance threshold is given then use 0.5 as the default.
        if self.cluster_distance_threshold is None:
            logging.info(
                "No 'cluster_distance_threshold' was given as input."
                + "A default value will thus be set."
            )
            self.cluster_distance_threshold = 0.5
            logging.info(
                "Determining clusters based on 'cluster_distance_threshold': "
                + str(self.cluster_distance_threshold),
            )

        logging.info(
            "Determining clusters based on 'cluster_distance_threshold': "
            + str(self.cluster_distance_threshold),
        )

        clustering_info = []
        for linkage_method in self.linkage_methods:
            clustering_info.append(
                self._perform_feature_clustering(
                    # X=_X,
                    X=X,
                    linkage_method=linkage_method,
                    pairwise_distances=pairwise_distances,
                )
            )
        #     if self._progress_bar is not None:
        #         # Update the progress bar
        #         self._progress_bar.update(1)
        # if self._progress_bar is not None:
        #     # Close/stop the progress bar
        #     self._progress_bar.close()

        # Save clustering results in sorted order according to the given
        # 'refit_criteria'
        self._df = pandas.DataFrame(data=clustering_info).sort_values(
            by=[self.refit_criteria],
            ascending=CLUSTERING_CRITERIA[self.refit_criteria],
        )
        # Set the best parameter values that have been found (based on a set
        # 'refit_criteria')
        self._set_best_parameters()
        # Continue if the best parameter values have been set correctly
        if self._idx is not None:
            self.clustered_association_matrix = X.iloc[self._idx, :].T.iloc[
                self._idx, :
            ]
            return self
        else:
            raise ValueError("Internal error. The indexing array '._idx' is None. ")

    def _check_vars(self) -> None:
        """
        Check that the given input arguments are valid.

        Raises:
            ValueError: If a given linkage method is not a valid linkage \
                method accepted as input.
            ValueError: If the given distance threshold is not in the range \
                [0, 1].
            ValueError: If the given criterion for determining the best \
                clustering method is not a valid criterion accepted as input.
        """
        for linkage_method in self.linkage_methods:
            if linkage_method not in LINKAGE_METHODS:
                raise ValueError(
                    "The given input argument 'linkage_methods' "
                    + f"{linkage_method} is not valid. "
                    + "Valid input values are: "
                    + " ".join(LINKAGE_METHODS)
                    + "."
                )
        if self.cluster_distance_threshold is not None:
            if (
                self.cluster_distance_threshold > 1
                or self.cluster_distance_threshold < 0
            ):
                raise ValueError(
                    "The gievn input argument "
                    + "'cluster_distance_threshold' "
                    + f"{self.cluster_distance_threshold} is not valid. "
                    + "The value should be in the range [0, 1]."
                )
        if self.refit_criteria not in list(CLUSTERING_CRITERIA.keys()):
            raise ValueError(
                "The given input argument 'refit_criteria' "
                + f"{self.refit_criteria} is not valid. "
                + "Valid input values are: "
                + " ".join(CLUSTERING_CRITERIA)
                + "."
            )

    def _set_best_parameters(self) -> None:
        """
        Set the parameters associated with the best clustering of highly
        associated features.

        Raises:
            ValueError: If no clustering has yet to be applied and no data \
                pertaining to the best clustering of features has been obtained.
        """
        if self._df is not None:
            row = self._df.iloc[0]
            self._idx = row["idx"]
            self._idx_to_cluster_array = row["idx_to_cluster_array"]
            self.linkage_method = row["linkage_method"]
            self.cophenetic_coefficient = row["cophenetic_coefficient"]
            self.linkage = row["linkage_object"]
            logging.info(
                "\nSetting the best clustering parameter values based on "
                + f"criterion: {self.refit_criteria}."
            )
            cluster_labels = numpy.unique(self._idx_to_cluster_array)
            logging.info(
                "Number of clusters                                   : "
                + f"{len(cluster_labels)}.\n"
                + "Cophenetic correlation coefficient (larger is better): "
                + f"{self.cophenetic_coefficient}.\n"
                + "Silhouette score coefficient       (larger is better): "
            )
        else:
            raise ValueError(
                "Internal error."
                + "The pandas dataframe '._df' is None. "
                + "Information associated with the best possible clustering "
                + "of features can thus not be set in the dataframe '._df'."
                + "Call method '.fit(X)' before calling "
                + "'._set_best_parameters()'"
            )

    def _perform_feature_clustering(
        self,
        # NOTE: Internal method. Assume 'X' is passed as a pandas dataframe
        X: pandas.DataFrame,
        linkage_method: str,
        pairwise_distances: numpy.ndarray,
    ) -> dict[str, typing.Any]:
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
        if not isinstance(X, pandas.DataFrame):
            raise ValueError(
                "The given input arugment 'X' should be of type "
                + f"'DataFrame' but is intead of type '{type(X)}'."
            )
        logging.info(f"\nApplying linkage method: {linkage_method}")
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
        idx = numpy.argsort(idx_to_cluster_array)

        # Compute the cophenetic correlation coefficient
        cophenetic_coef, _ = hac.cophenet(Z=linkage, Y=pairwise_distances)
        cluster_labels = numpy.unique(idx_to_cluster_array)
        logging.info(
            "Number of clusters                                   : "
            + f"{len(cluster_labels)}.\n"
            + "Cophenetic correlation coefficient (larger is better): "
            + f"{cophenetic_coef}.\n"
        )

        # Return data associated with the obtained clustering so we subsequently
        # can determine the best approach
        return {
            "linkage_method": linkage_method,
            "cophenetic_coefficient": cophenetic_coef,
            "idx_to_cluster_array": idx_to_cluster_array,
            "idx": idx,
            "linkage_object": linkage,
        }


class Cluster:
    def __init__(self, feature_names: list[str]) -> None:
        _feature_names: dict[str, int] = {label: 0 for label in feature_names}
        self.feature_names = OrderedDict(
            sorted(_feature_names.items(), key=itemgetter(1))
        )

    def get_feature(self) -> str:
        return self._first(self.feature_names)

    def update_counter(self, feature_name: str) -> None:
        self.feature_names[feature_name] += 1
        # Update the odered dictionary with new counts
        self.feature_names = OrderedDict(
            sorted(self.feature_names.items(), key=itemgetter(1))
        )

    def _first(self, collection: OrderedDict) -> str:
        """
        Return the first element from an ordered collection or an arbitrary
        element from an unordered collection.

        Raise StopIteration if the collection is empty.
        """
        return next(iter(collection))


class ClusterSampler:
    def __init__(self, feature_clusters: list[list[str]]) -> None:
        self.clusters: dict[str, Cluster] = {}
        counter = 0
        for feature_names in feature_clusters:
            self.clusters[f"cluster{counter}"] = Cluster(
                feature_names=feature_names
            )
            counter += 1

    def sample_feature_subset(self) -> list[str]:
        feature_subset: list[str] = []
        for cluster_label in self.clusters:
            feature_name = self.clusters[cluster_label].get_feature()
            feature_subset.append(feature_name)
            self.clusters[cluster_label].update_counter(feature_name)
        return feature_subset
