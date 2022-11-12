"""This file contains methods for plotting results and other information that
have been obtained through the application of the ShapFire method."""

import typing

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.metrics import auc

# Define the main color palette to use for plots and other illustations
MAIN_COLOR_PALETTE: dict[str, str] = {
    "background": "#feffff",
    "selected": "#25c5da",
    "rejected": "#010105",
    "secondary": "#c53e6e",
    "tertiary": "#3e77bf",
    "overlay": "#9c9d9d",
    "grid": "#d2d2d2",
}

# Define the main matplotlib and seaborn plotting settings
DEFAULT_PLOTTING_SETTINGS: dict[str, typing.Any] = {
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
    def __init__(self, shapfire) -> None:  # type: ignore
        # Class vars corresponding to input args
        self.shapfire = shapfire  # noqa

        # Internal vars for easy access to data associated with the importance
        # ranking of features
        self._data: typing.Union[None, dict[str, typing.Any]] = None
        self._is_jointplot: bool = False
        self._groupby: typing.Union[None, str] = None

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
        plot_type = "stripplot"
        # Define the default plotting options
        PLOT_IMPORTANCE_OPTIONS: dict[str, typing.Any] = {
            # Do not allow violinplot. The elements will be squished too
            # much and result in an awful representation of the data
            "stripplot": {
                "func": sns.stripplot,
                "xargs": {"dodge": True, "alpha": 1.0, "ax": ax, "marker": "o"},
            },
        }

        # Validate given input arguments
        if not isinstance(groupby, str):
            raise TypeError(
                "The given input argument 'groupby' should be of type "
                + f"'str' but an argument of type '{type(groupby)}' was given."
            )
        else:
            GROUBPBY_OPTIONS = ["feature", "cluster"]
            if not groupby.strip().lower() in GROUBPBY_OPTIONS:
                raise ValueError(
                    "The given input argument 'groupby' should be one "
                    + f"of the following options: {', '.join(GROUBPBY_OPTIONS)}"
                    + f" but an argument '{groupby}' was given."
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
        _apply_default_styling(rcParams)

        # Prepare the appropriate data for plotting
        _groupby = groupby.strip().lower()
        if self._data is None or _groupby != self._groupby:
            self._data = self._prepare_data(groupby=_groupby)
            self._groupby = _groupby

        # Unpack all necessary data for plotting
        df = self._data["df"]
        feature_ordering = self._data["feature_ordering"]
        feature_colors = self._data["feature_colors"]
        cluster_labels = self._data["cluster_labels"]

        # Determine the searborn function to use for plotting and set function
        # arguments that should be passed to the plotting function
        args = {
            "x": "ranked_difference",
            "y": df.index,
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

        x_max = df["ranked_difference"].max()
        df["ranked_difference"].min()

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
                    x_offset=x_max * 1.075,
                )
        elif groupby.strip().lower() == "feature":
            if with_overlay is True:
                # Add two alternating gray-scale colors for better seperation
                # of the plotted data. By default do not add text information
                # about the clusters each feature belong to. For this purpose,
                # the groupby = "cluster" should be chosen
                self._add_feature_overlays(ax=ax, cluster_labels=cluster_labels)
        else:
            raise ValueError(
                "The given input argument 'groupby' should have value "
                + f"'feature' or 'cluster' but value '{groupby}' was given."
            )

        # Indicate the cut-off threshold
        ax.axvline(
            self.shapfire.threshold_finder.lower_threshold,
            linestyle="--",
            lw=2.0,
            color=MAIN_COLOR_PALETTE["tertiary"],
        )

        # Add a legend to the figure indicating which feautres have been
        # selected and which have been rejected
        custom_lines = [
            Line2D(
                [0],
                [0],
                color=MAIN_COLOR_PALETTE["tertiary"],
                lw=2.25,
                linestyle="--",
            ),
            Line2D(
                [0],
                [0],
                color=MAIN_COLOR_PALETTE["selected"],
                lw=2.25,
            ),
            Line2D(
                [0],
                [0],
                color=MAIN_COLOR_PALETTE["rejected"],
                lw=2.25,
            ),
        ]
        ax.legend(
            custom_lines,
            ["Threshold", "Selected", "Rejected"],
            loc="upper right",
            fontsize=fontsize + 1,
            handlelength=2.75,
        )

        # Make changes related to figure size, title, x-axis labels + ticks,
        # y-axis labels + and ticks, etc.
        ax.set_title(
            "Importance ranking & selected features",
            fontsize=fontsize + 1,
            pad=20,
        )
        ax.set_xlabel("Ranked difference", fontsize=fontsize + 1)
        ax.set_ylabel("Feature name", fontsize=fontsize + 1)

        # Set x and y-axis ticks
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=fontsize)
        ax.set_zorder(1)

        # Set figure size
        fig = _set_figure_size(
            figsize=figsize,
            fig=fig,
            num_all_features=len(numpy.unique(df.index.to_list())),
        )
        return fig, ax

    def _organize_data_for_plotting(
        self,
        feature_ranking_df: pandas.DataFrame,
        df: pandas.DataFrame,
        groupby: str,
    ) -> tuple[pandas.DataFrame, list[str]]:
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
            names = numpy.unique(df["feature_name"]).tolist()
            template_dict = {n: numpy.nan for n in names}
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
                new_df = pandas.DataFrame(data=data).reindex(indexing, axis=1)
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
                new_df = pandas.DataFrame(data=data).reindex(indexing, axis=1)
                return new_df, list(new_df.columns.values)
            else:
                raise ValueError(
                    "The given input argument 'groupby' should have value "
                    + f"'feature' or 'cluster' but a value '{groupby}' was "
                    + "given."
                )

    # def _prepare_data(
    #     self,
    #     groupby: str,
    # ) -> dict[str, typing.Any]:
    #     feature_ranking_df = self.shapfire.ranked_differences

    #     if groupby.strip().lower() == "cluster":
    #         _feature_ranking_df = feature_ranking_df.copy()
    #         cluster_ordering = (
    #             _feature_ranking_df.groupby(by=["cluster_label"])
    #             .head(1)["cluster_label"]
    #             .values
    #         )
    #         _feature_ranking_df["cluster_label"] = pandas.Categorical(
    #             _feature_ranking_df["cluster_label"].values,
    #             categories=cluster_ordering,
    #         )
    #         _feature_ranking_df.sort_values(
    #             by=["cluster_label", "ranked_difference"],
    #             ascending=[True, True],
    #             inplace=True,
    #         )
    #     elif groupby.strip().lower() == "feature":
    #         _feature_ranking_df = feature_ranking_df.copy()
    #     else:
    #         raise ValueError(
    #             "The given input argument 'groupby' should have value "
    #             + f"'feature' or 'cluster' but value '{groupby}' was given."
    #         )

    #     # Extract a list of clusters that each selected feature is associated
    #     # with
    #     cluster_labels = _feature_ranking_df["cluster_label"].values.tolist()
    #     all_feature_names = _feature_ranking_df.index.to_list()
    #     selected_features = self.shapfire.selected_features

    #     # Set colors for each selected/rejected feature
    #     feature_colors = {}
    #     for feature_name in all_feature_names:
    #         if feature_name in selected_features:
    #             feature_colors[feature_name] = MAIN_COLOR_PALETTE["selected"]
    #         else:
    #             feature_colors[feature_name] = MAIN_COLOR_PALETTE["rejected"]

    #     x = self.shapfire.ranked_differences["ranked_difference"]
    #     y = self.shapfire.ranked_differences.index
    #     num_all_features = numpy.unique(y).shape[0]

    #     _y = y.to_frame()
    #     _y["feature_name"] = pandas.Categorical(
    #         _y.index,
    #         all_feature_names,
    #     )
    #     _y.sort_values(by=["feature_name"], inplace=True)

    #     return {
    #         "df": self.shapfire.ranked_differences,
    #         "feature_ordering": all_feature_names,
    #         "feature_colors": feature_colors,
    #         "selected_features": selected_features,
    #         "all_feature_names": all_feature_names,
    #         "cluster_labels": cluster_labels,
    #         # Other fields...
    #         "x": x,
    #         "y": y,
    #         "_y": _y,
    #         "num_all_features": num_all_features,
    #     }

    def _prepare_data(
        self,
        groupby: str,
    ) -> dict[str, typing.Any]:
        # Calculate normalized SHAP feature importance scores
        df = self.shapfire._calculate_normalized_shap_feature_importance()

        # Extract feature ranking, i.e., determine the ordering of the
        # features on the x-axis based on the computed median
        # 'normalized_feature_importance'
        feature_ranking_df = self.shapfire._pick_top_k_from_clusters(
            df=df,
            top_k=None,
        )

        # TEMP!
        self._temp_feature_ranking_df = feature_ranking_df

        if groupby.strip().lower() == "cluster":
            _feature_ranking_df = feature_ranking_df.copy()
            cluster_ordering = (
                _feature_ranking_df.reset_index(level=["cluster"])
                .groupby(by=["cluster"])
                .head(1)["cluster"]
                .values
            )
            _feature_ranking_df = _feature_ranking_df.reset_index(
                level=["cluster"]
            )
            _feature_ranking_df["cluster"] = pandas.Categorical(
                _feature_ranking_df["cluster"].values,
                categories=cluster_ordering,
            )
            _feature_ranking_df.sort_values(
                by=["cluster", "normalized_feature_importance"],
                ascending=[True, False],
                inplace=True,
            )
        elif groupby.strip().lower() == "feature":
            _feature_ranking_df = feature_ranking_df.copy()
            _feature_ranking_df = _feature_ranking_df.reset_index(
                level=["cluster"]
            )
        else:
            raise ValueError(
                "The given input argument 'groupby' should have value "
                + f"'feature' or 'cluster' but value '{groupby}' was given."
            )

        # Extract a list of clusters that each selected feature is associated
        # with
        cluster_labels = _feature_ranking_df["cluster"].values.tolist()
        # all_features = feature_ranking_df.reset_index(level=["cluster"])
        all_feature_names = _feature_ranking_df.index.values
        
        # Select best features from each cluster
        # selected_features = feature_ranking_df.groupby(by=["cluster"]).head(1)

        # TODO: 
        # Select all features from each cluster
        selected_features = feature_ranking_df.groupby(by=["cluster"]).head(1)

        # Discard features with 0 importance. A selected feature should
        # not have 0 importance!
        selected_features = selected_features[
            selected_features["normalized_feature_importance"] > 0
        ]

        # Further filtering based on a cutoff value
        cut_value = self.shapfire._find_cutoff(df=selected_features)
        print("SHAP importance cutoff value: ", cut_value)
        selected_features = selected_features[
            selected_features["normalized_feature_importance"] >=  cut_value
        ]

        selected_feature_names = [v[0] for v in selected_features.index.values]

        plotting_df, feature_ordering = self._organize_data_for_plotting(
            df=df,
            feature_ranking_df=_feature_ranking_df,
            groupby=groupby,
        )

        # Set colors for each selected/rejected feature
        feature_colors = {}
        for feature_name in all_feature_names:
            if feature_name in selected_feature_names:
                feature_colors[feature_name] = MAIN_COLOR_PALETTE["selected"]
            else:
                feature_colors[feature_name] = MAIN_COLOR_PALETTE["rejected"]

        # Generate information related to feature subsets
        # Determine the ordering of the different evaluated feature
        # subsets on the y-axis based on the score associated with the
        # feature subset and that has been obtained
        test_folds = (
            df.groupby("test_fold")
            .agg("median")
            .sort_values(by=["score"], ascending=True)
            .index
        )

        x_name = "test_fold_rank"

        for counter, value in enumerate(test_folds):
            df.loc[df["test_fold"] == value, x_name] = counter + 1

        # Extract data necessary for plotting feature subsets
        x, y = df[x_name], df["feature_name"]
        num_all_features = numpy.unique(df["feature_name"]).shape[0]
        num_all_feature_subsets = numpy.unique(df[x_name]).shape[0]
        best_score = df.loc[df[x_name] == df[x_name].max(), "score"].iat[0]
        worst_score = df.loc[df[x_name] == df[x_name].min(), "score"].iat[0]

        _y = y.to_frame()
        _y["feature_name"] = pandas.Categorical(
            _y["feature_name"],
            all_feature_names,
        )
        _y.sort_values(by=["feature_name"], inplace=True)

        return {
            # Main importance plot fields...
            "df": df,
            "feature_ordering": feature_ordering,
            "feature_colors": feature_colors,
            "selected_features": selected_features,
            # "all_features": all_features,
            "all_features": None,
            "all_feature_names": all_feature_names,
            "cluster_labels": cluster_labels,
            "plotting_df": plotting_df,
            # Other fields...
            "x": x,
            "y": y,
            "_y": _y,
            "num_all_features": num_all_features,
            "num_all_feature_subsets": num_all_feature_subsets,
            "best_score": best_score,
            "worst_score": worst_score,
        }


    def _add_cluster_overlays(
        self,
        ax: Axes,
        cluster_labels: list[str],
        fontsize: int = 10,
        with_text: bool = True,
        with_overlay: bool = True,
        x_offset: float = 0,
    ) -> None:
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
        ax: Axes,
        cluster_labels: list[str],
    ) -> None:
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


    def ceil5(self, x: typing.Union[float, int]) -> int:
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
        return int(base * numpy.ceil(x / base))

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
        # Define the default plotting options
        PLOT_IMPORTANCE_OPTIONS: dict[str, typing.Any] = {
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
            self._data = self._prepare_data(groupby=_groupby)
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

        # TODO: Remove
        # print()
        # print(df)
        # print()

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
        # 0.00 - 0.025
        # x_max + 0.025

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
        elif groupby.strip().lower() == "feature":
            if with_overlay is True:
                # Add two alternating gray-scale colors for better seperation
                # of the plotted data. By default do not add text information
                # about the clusters each feature belong to. For this purpose,
                # the groupby = "cluster" should be chosen
                self._add_feature_overlays(ax=ax, cluster_labels=cluster_labels)
        else:
            raise ValueError(
                "The given input argument 'groupby' should have value "
                + f"'feature' or 'cluster' but value '{groupby}' was given."
            )


        # Indicate the cut-off threshold
        ax.axvline(
            # self.shapfire.threshold_finder.lower_threshold,
            self.shapfire.cut_value,
            linestyle="--",
            lw=2.25,
            color=MAIN_COLOR_PALETTE["tertiary"],
        )

        # Add a legend to the figure indicating which feautres have been
        # selected and which have been rejected
        custom_lines = [
            Line2D(
                [0],
                [0],
                color=MAIN_COLOR_PALETTE["tertiary"],
                lw=2.25,
                linestyle="--",
            ),
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
            ["Threshold", "Selected", "Rejected"],
            loc="lower right",
            fontsize=fontsize + 1,
        )

        # Make changes related to figure size, title, x-axis labels + ticks,
        # y-axis labels + and ticks, etc.
        ax.set_title(
            "ShapFire importance ranking and selected features",
            fontsize=fontsize + 1,
            # pad=20,
        )
        ax.set_xlabel(
            "Normalized SHAP feature importance", fontsize=fontsize + 1
        )
        
        # Only display y-axis label if it is plotted alone
        # if self._is_jointplot is False:
        #     ax.set_ylabel("Feature name", fontsize=fontsize + 1)
        # else:
        #     ax.set_ylabel(None)
        # ax.tick_params(axis="both", which="major", labelsize=fontsize)
        # ax.tick_params(axis="both", which="minor", labelsize=fontsize)
        # ax.set_zorder(1)

        # Set figure size
        print("num all features: ", len(numpy.unique(df.columns.to_list())))
        print(df)

        fig = _set_figure_size(
            figsize=figsize,
            fig=fig,
            num_all_features=len(numpy.unique(df.index.to_list())),
        )

        return fig, ax

    def plot_evaluated_feature_subsets(
        self,
        groupby: str = "cluster",
        rcParams: typing.Union[None, dict[str, str]] = None,
        figsize: typing.Union[None, tuple[float, float]] = None,
        fontsize: int = 10,
        marker: str = "o",
        markersize: int = 5,
        with_text: bool = True,
        with_overlay: bool = True,
        axes: typing.Union[None, list[Axes]] = None,
    ) -> tuple[Figure, list[Axes]]:
        # Validate given input arguments
        if isinstance(axes, list):
            if len(axes) >= 2:
                ax0, ax1 = axes[0], axes[1]
                fig = ax0.get_figure()
            else:
                raise ValueError(
                    "The given input argument 'axes' should have length "
                    + f">= 2 but the given list instead had length {len(axes)}."
                )
        elif axes is None:
            fig, axes = plt.subplots(
                nrows=1,
                ncols=2,
                sharey=True,
                sharex=False,
                gridspec_kw={"width_ratios": [5, 1]},
            )
            ax0, ax1 = axes[0], axes[1]
        else:
            raise ValueError(
                "The given input argument 'axes' should be of type 'list' "
                + f" or 'None' but is instead of type {type(axes)}."
            )

        # Apply styling to the plot elements
        self._apply_styling(rcParams=rcParams)

        # Prepare the appropriate data for plotting
        _groupby = groupby.strip().lower()
        if self._data is None or _groupby != self._groupby:
            self._data = self._prepare_data(groupby=_groupby)
            self._groupby = _groupby

        # Only adjust the size of the figure here if it is not being plotted
        # with together with other types of plots in the current figure
        if self._is_jointplot is False:
            fig, figsize = _set_figure_size(
                figsize=figsize,
                fig=fig,
                num_all_features=len(
                    numpy.unique(self._data["df"]["feature_name"])
                ),
            )

        # Unpack all necessary data for plotting
        all_feature_names = self._data["all_feature_names"]
        df = self._data["df"]
        x = self._data["x"]
        y = self._data["y"]
        _y = self._data["_y"]
        self._data["num_all_features"]
        self._data["num_all_feature_subsets"]
        best_score = self._data["best_score"]
        worst_score = self._data["worst_score"]
        cluster_labels = self._data["cluster_labels"]

        # Create stripplot
        sns.stripplot(
            x=x,
            y=y,
            s=markersize,
            marker=marker,
            linewidth=1.5,
            color="black",
            ec="black",
            fc="none",
            jitter=False,
            order=all_feature_names,
            ax=ax0,
        )
        sns.despine(
            ax=ax0,
            top=True,
            bottom=True,
            right=True,
            left=True,
            offset=None,
            trim=False,
        )

        # Create an associated vertical histogram displaying the number of times
        # a certain feature has been a part of an evaluated feature subset
        sns.histplot(
            y=_y["feature_name"],
            discrete=True,
            linewidth=1.5,
            ax=ax1,
            color="black",
            shrink=0.95,
        )
        sns.despine(
            ax=ax1,
            top=True,
            bottom=True,
            right=True,
            left=True,
            offset=None,
            trim=False,
        )

        # Extract the histogram max and min values so the axis can be adjusted
        # accordingly
        ymax = self.ceil5(df["feature_name"].value_counts().max())
        ax1.set_xticks([0, ymax])
        ax1.set_xlim([0, ymax + 1])

        # Set axis ax1 tick size
        ax1.tick_params(axis="both", which="major", labelsize=fontsize)
        ax1.tick_params(axis="both", which="minor", labelsize=fontsize)

        # Set axis ax0 tick size
        ax0.tick_params(axis="both", which="major", labelsize=fontsize)
        ax0.tick_params(axis="both", which="minor", labelsize=fontsize)

        # Add additional plot overlays depending on how features should be
        # grouped and displayed in the plot
        if groupby.strip().lower() == "cluster":
            if with_text is True or with_overlay is True:
                # Add two alternating gray-scale colors for grouping features
                # based on the cluster they each belong to. Also, add text
                # information about the cluster each feature belongs to
                # - Overlays on axis: ax0
                self._add_cluster_overlays(
                    ax=ax0,
                    cluster_labels=cluster_labels,
                    # Do not display text. The placement will not wrong!
                    with_text=False,
                    with_overlay=with_overlay,
                )
                # - Overlays on axis: ax1
                self._add_cluster_overlays(
                    ax=ax1,
                    cluster_labels=cluster_labels,
                    with_text=with_text,
                    with_overlay=with_overlay,
                    x_offset=ymax * 1.10,
                )
        elif groupby.strip().lower() == "feature":
            if with_overlay is True:
                # Add two alternating gray-scale colors for better seperation
                # of the plotted data. By default do not add text information
                # about the clusters each feature belong to. For this purpose,
                # the groupby = "cluster" should be chosen.
                # - Overlays on axis: ax0
                self._add_feature_overlays(
                    ax=ax0,
                    cluster_labels=cluster_labels,
                )
                # - Overlays on axis: ax1
                self._add_feature_overlays(
                    ax=ax1,
                    cluster_labels=cluster_labels,
                )
        else:
            raise ValueError(
                "The given input argument 'groupby' should have value "
                + f"'feature' or 'cluster' but value '{groupby}' was given."
            )

        subsets = 0
        # Add vertical lines to visually indicate feautre subsets
        for index, _df in df.groupby("test_fold"):
            ax0.plot(
                [index for _ in ax0.get_yticks()],
                ax0.get_yticks(),
                color="black",
                alpha=0.125,
            )
            subsets += 1

        # Make changes related to figure size, title, x-axis labels + ticks,
        # y-axis labels + and ticks, etc.

        # Set plot labels
        ax0.set_xlabel(
            "Feature subset ordered by score",
            fontsize=fontsize + 1,
        )
        ax0.set_ylabel("Feature name", fontsize=fontsize + 1)
        ax0.set_title(
            "Evaluated feature subsets",
            fontsize=fontsize + 1,
            pad=20,
        )
        ax1.set_xlabel(
            "Counts",
            fontsize=fontsize + 1,
        )

        # Add major axis ticks and remove minor axis ticks
        ax0.set_xticks(
            [0, subsets], [f"{worst_score:.3f}", f"{best_score:.3f}"]
        )
        ax0.set_xticks([], minor=True)
        # Disable grids in the two plots. They conflict with the result of
        # using plotting method 'axvline'
        ax0.grid(False)
        ax1.grid(False)

        # Only adjust the size of the figure here if it is not being plotted
        # with together with other types of plots in the current figure
        if self._is_jointplot is False:
            fig, figsize = _set_figure_size(
                figsize=figsize,
                fig=fig,
                num_all_features=len(numpy.unique(df["feature_name"])),
            )

        return fig, [ax0, ax1]

    def _apply_styling(
        self,
        rcParams: typing.Union[None, dict[str, typing.Any]] = None,
    ) -> None:
        # Apply default styling to the generated plots
        sns.set_theme(style="whitegrid")
        if rcParams is not None:
            mpl.rcParams.update(rcParams)
        else:
            sns.set_context("paper", rc=DEFAULT_PLOTTING_SETTINGS)

def plot_roc_curve(
    df: pandas.DataFrame,
    figsize: tuple[float, float] = (8, 4),
    plot_all_curves: bool = True,
    ax: typing.Union[None, Axes] = None,
    **kwargs: dict[str, typing.Any],
) -> tuple[Figure, Axes]:
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
    mean_fpr = numpy.linspace(start=0, stop=1, num=100)
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
        interp_tpr = numpy.interp(x=mean_fpr, xp=fpr, fp=tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

    # Plot mean of ROC lines
    mean_tpr = numpy.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = numpy.std(aucs, ddof=1)
    ax.plot(
        mean_fpr,
        mean_tpr,
        lw=mean_linewidth,
        alpha=mean_alpha,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    )

    # Fill between ROC lines
    std_tpr = numpy.std(tprs, axis=0)
    tprs_upper = numpy.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = numpy.maximum(mean_tpr - std_tpr, 0)
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
    ax: Axes,
    fpr: numpy.ndarray,
    tpr: numpy.ndarray,
    roc_auc: float,
    fold: float,
    line_linewidth: float = 1.25,
    line_alpha: float = 0.30,
) -> Axes:
    ax.plot(
        fpr,
        tpr,
        lw=line_linewidth,
        alpha=line_alpha,
        label=f"Fold {fold}. ROC (AUC = {roc_auc:.2f})",
    )
    return ax


def _add_dummy_legend_entries(
    total_lines: int,
    lines_per_inch: float,
    figsize: tuple[float, float],
    ax: Axes,
) -> int:
    lines_per_inch = 4
    total_space = figsize[1] * lines_per_inch
    ncols = int(numpy.ceil(total_lines / total_space))
    dummy_lines = int(total_space - (total_lines % total_space))
    # Add additional dummy legend entries to fill empty space
    # in the displayed legend
    for _ in range(dummy_lines):
        ax.plot([], [], color="black", lw=0, alpha=0, label=" ")
    return ncols


def _apply_default_styling(
    rcParams: typing.Union[None, dict[str, typing.Any]] = None,
) -> None:
    # Apply default styling to the generated plots
    sns.set_theme(style="whitegrid")
    if rcParams is not None:
        mpl.rcParams.update(rcParams)
    else:
        sns.set_context("paper", rc=DEFAULT_PLOTTING_SETTINGS)


def _set_figure_size(
    figsize: typing.Union[None, tuple[float, float]],
    fig: Figure,
    num_all_subsets: typing.Union[None, float, int] = None,
    num_all_features: typing.Union[None, float, int] = None,
    default_figsize: typing.Union[None, tuple[float, float]] = None,
) -> tuple[Figure, tuple[float, float]]:
    if figsize is None:
        if num_all_features is not None:
            height = 2.5 * numpy.maximum(numpy.rint(num_all_features / 10), 1)
            if num_all_subsets is not None:
                width = 6 * numpy.rint(num_all_subsets / 10)
            else:
                width = 1 * height
            if default_figsize is not None:
                figsize = (
                    numpy.maximum(width, default_figsize[0]),
                    numpy.maximum(height, default_figsize[1]),
                )
            else:
                figsize = (width, height)
            print("Setting figure size 1: ", figsize)
            fig.set_size_inches(figsize)
        else:
            raise ValueError(
                "The given input argument 'num_all_features' and "
                + "'figsize' is None. If 'figsize' is None, then "
                + "'num_all_features' needs to be an integer value "
                + "larger than 0."
            )
    else:
        fig.set_size_inches(figsize)
        print("Setting figure size 2: ", figsize)
    return fig, figsize
