"""This module contains a collection of general constants and functions that are
used by several other functions and classes in the ShapFire library."""


import typing

import numpy
import pandas
import scipy.stats as stats

REPLACE: str = "replace"
"""The default string value used to indicate that NaN or None values should be \
replaced with another given value."""  # pylint: disable=W0105

DROP: str = "drop"
"""The default string value used to indicate that samples associated with a \
dataset (X) and target variable (y) should be dropped if NaN or None values \
are contained in a sample.
"""  # pylint: disable=W0105

DROP_SAMPLES: str = "drop_samples"
"""The default string value used to indicate that a sample (row) in a dataset \
(X) should be dropped if it contains NaN or None values.
"""  # pylint: disable=W0105

DROP_FEATURES: str = "drop_features"
"""The default string value used to indicate that a feature (column) in a \
dataset (X) should be dropped if it contains NaN or None values.
"""  # pylint: disable=W0105

SKIP: str = "skip"
"""The default string value used to indicate that a value should be skipped \
whenever a NaN or None value is encountered.
"""  # pylint: disable=W0105

DEFAULT_REPLACE_VALUE: float = 0.0
"""The default value that NaN or None values are replaced with.
"""  # pylint: disable=W0105

DEFAULT_RANDOM_SEED: int = 123
"""The default random seed used across modules."""  # pylint: disable=W0105


def remove_incomplete_samples(
    x: typing.Union[list, numpy.ndarray],
    y: typing.Union[list, numpy.ndarray],
) -> typing.Union[tuple[list, list], tuple[numpy.ndarray, numpy.ndarray]]:
    x = [v if v is not None else numpy.nan for v in x]
    y = [v if v is not None else numpy.nan for v in y]
    arr = numpy.array([x, y]).transpose()
    arr = arr[~numpy.isnan(arr).any(axis=1)].transpose()
    if isinstance(x, list):
        return arr[0].tolist(), arr[1].tolist()
    else:
        return arr[0], arr[1]


def replace_nan_with_value(
    x: numpy.ndarray, y: numpy.ndarray, value: float
) -> tuple[numpy.ndarray, numpy.ndarray]:
    x = numpy.array([v if v == v and v is not None else value for v in x])
    y = numpy.array([v if v == v and v is not None else value for v in y])
    return x, y


def convert(
    data: typing.Union[numpy.ndarray, pandas.Series, pandas.DataFrame, list],
    to: str,
    copy: bool = True,
) -> typing.Union[numpy.ndarray, pandas.Series, pandas.DataFrame, list]:
    converted: typing.Union[
        None, numpy.ndarray, pandas.Series, pandas.DataFrame, list
    ] = None
    if to.strip().lower() == "array":
        if isinstance(data, numpy.ndarray):
            converted = data.copy() if copy else data
        elif isinstance(data, pandas.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = numpy.array(data)
        elif isinstance(data, pandas.DataFrame):
            converted = data.values()
    elif to.strip().lower() == "list":
        if isinstance(data, list):
            converted = data.copy() if copy else data
        elif isinstance(data, pandas.Series):
            converted = data.values.tolist()
        elif isinstance(data, numpy.ndarray):
            converted = data.tolist()
    elif to.strip().lower() == "dataframe":
        if isinstance(data, pandas.DataFrame):
            converted = data.copy(deep=True) if copy else data
        elif isinstance(data, numpy.ndarray):
            converted = pandas.DataFrame(data)
    else:
        raise ValueError(f"Unknown data conversion: {to}")
    if converted is None:
        raise TypeError(
            f"Cannot handle data conversion of type: {type(data)} to {to}"
        )
    else:
        return converted


def cramers_v(
    # TODO: Allow 'x' to be an numpy.ndarray
    x: pandas.Series,
    # TODO: Allow 'y' to be an numpy.ndarray
    y: pandas.Series,
    bias_correction: bool = True,
    nan_strategy: str = REPLACE,
    nan_replace_value: float = DEFAULT_REPLACE_VALUE,
) -> float:
    if nan_strategy == REPLACE:
        x, y = replace_nan_with_value(
            x=x,
            y=y,
            value=nan_replace_value,
        )
    elif nan_strategy == DROP:
        x, y = remove_incomplete_samples(x=x, y=y)
    confusion_matrix = pandas.crosstab(index=x, columns=y)
    chi2, _, _, _ = stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    if bias_correction:
        phi2corr = numpy.maximum(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        if numpy.minimum((kcorr - 1), (rcorr - 1)) == 0:
            print(
                "Unable to calculate Cramer's V using bias correction. "
                + "Consider using bias_correction=False"
            )
            return numpy.nan
        else:
            return numpy.sqrt(
                phi2corr / numpy.minimum((kcorr - 1), (rcorr - 1))
            )
    else:
        if numpy.minimum(k - 1, r - 1) == 0:
            return numpy.nan
        else:
            return numpy.sqrt(phi2 / numpy.minimum(k - 1, r - 1))


def correlation_ratio(
    # TODO: Allow 'categories' to be an numpy.ndarray
    categories: pandas.Series,
    # TODO: Allow 'measurements' to be an numpy.ndarray
    measurements: pandas.Series,
    nan_strategy: str = REPLACE,
    nan_replace_value: float = DEFAULT_REPLACE_VALUE,
) -> float:
    if nan_strategy == REPLACE:
        categories, measurements = replace_nan_with_value(
            x=categories,
            y=measurements,
            value=nan_replace_value,
        )
    elif nan_strategy == DROP:
        categories, measurements = remove_incomplete_samples(
            x=categories, y=measurements
        )
    categories = convert(data=categories, to="array")
    measurements = convert(data=measurements, to="array")
    fcat, _ = pandas.factorize(categories)
    cat_num = numpy.max(fcat) + 1
    y_avg_array = numpy.zeros(cat_num)
    n_array = numpy.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[numpy.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = numpy.average(cat_measures)
    y_total_avg = numpy.sum(numpy.multiply(y_avg_array, n_array)) / numpy.sum(
        n_array
    )
    numerator = numpy.sum(
        numpy.multiply(
            n_array,
            numpy.power(
                numpy.subtract(
                    y_avg_array,
                    y_total_avg,
                ),
                2,
            ),
        )
    )
    denominator = numpy.sum(
        numpy.power(
            numpy.subtract(
                measurements,
                y_total_avg,
            ),
            2,
        )
    )
    if numerator == 0:
        eta = 0.0
    else:
        eta = numpy.sqrt(numerator / denominator)
    return eta


def associations(
    X: pandas.DataFrame,
    nan_strategy: str = DROP_SAMPLES,
    nan_replace_value: float = DEFAULT_REPLACE_VALUE,
) -> pandas.DataFrame:
    """
    Calculate pairwise measures of association/correlation between numerical and
    categorical features in a given dataset. Numerical-numerical association is
    measured through Spearman's correlation coefficient, numerical-categorical
    association is measured through the correlation ratio and categorical-
    categorical association is measured through Cramer's V.

    Args:
        X: The input dataset that is assumed to contain features (columns) and \
            corresponding observations (rows).
        nan_strategy: The action to take in case the input dataset contains \
            NaN or None values. Defaults to DROP_SAMPLES.
        nan_replace_value: In case the :code:`nan_strategy` is \
            :const:`shapfire.utils.REPLACE`, then this argument \
            determines the value which NaN or None values are replaced by. \
            Defaults to :const:`shapfire.utils.DEFAULT_REPLACE_VALUE`.

    Raises:
        ValueError: If the number of `category` and `float` features (columns) \
            in the pandas dataframe do not add up to the total number of \
            features (columns) contained in the dataframe.

    Returns:
        A symmetric pandas dataframe that contains all pariwise feature \
        correlation/association values.
    """

    # Extract dataframe column labels
    columns = X.columns

    # Apply a strategy for handling NaN values in the given data
    if nan_strategy == REPLACE:
        _X = X.fillna(value=nan_replace_value, inplace=False)
    elif nan_strategy == DROP_SAMPLES:
        _X = X.dropna(axis=0, inplace=False)
    elif nan_strategy == DROP_FEATURES:
        _X = X.dropna(axis=1, inplace=False)
    else:
        _X = X.copy()

    # Identify categorical features and columns
    cat_columns = _X.select_dtypes(include=["category"]).columns
    # Identify numerical features and columns
    num_columns = _X.select_dtypes(include=["float"]).columns
    if len(cat_columns) + len(num_columns) != _X.shape[1]:
        # Make sure that columns are either of type 'category' or type 'float'
        raise ValueError(
            "The number of categorical and numerical features (columns) in "
            + "the dataframe do not add up to the total number of features "
            + "(columns) that are actually contained in the dataframe. Make "
            + "sure the data contained in the columns are either of 'dtype' "
            + "'float' or 'category'."
        )

    # Create dataframe for storing associations values
    c = pandas.DataFrame(index=columns, columns=columns)

    # Find single-value columns
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
                            nan_strategy=SKIP,
                        )
                        ij, ji = cell, cell
                    else:
                        cell = correlation_ratio(
                            _X[columns[i]],
                            _X[columns[j]],
                            nan_strategy=SKIP,
                        )
                        ij, ji = cell, cell
                else:
                    if columns[j] in cat_columns:
                        cell = correlation_ratio(
                            _X[columns[j]],
                            _X[columns[i]],
                            nan_strategy=SKIP,
                        )
                        ij, ji = cell, cell
                    else:
                        cell, _ = stats.spearmanr(
                            _X[columns[i]],
                            _X[columns[j]],
                        )
                        ij, ji = cell, cell
                c.loc[columns[i], columns[j]] = (
                    ij if not numpy.isnan(ij) and abs(ij) < numpy.inf else 0.0
                )
                c.loc[columns[j], columns[i]] = (
                    ji if not numpy.isnan(ji) and abs(ji) < numpy.inf else 0.0
                )
    c.fillna(value=numpy.nan, inplace=True)
    return c


class ThresholdFinder:
    def __init__(
        self,
        random_seed: int,
        ncols: int,
        n_batches: int = 250,
        n_samples: int = 1000,
    ) -> None:
        self.ncols = ncols
        self.n_batches = n_batches
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.data: typing.Union[None, list[float]] = None

        self.lower_threshold: typing.Union[None, float] = None
        self.iqr: typing.Union[None, float] = None
        self.upper_threshold: typing.Union[None, float] = None

    def fit(self) -> dict[str, float]:
        self.data = self.estimate_ranking_distribution()
        a, b, c = self.find_quartiles(data=self.data)

        self.lower_threshold = a
        self.iqr = b
        self.upper_threshold = c
        return {
            "lower_threshold": a,
            "iqr": b,
            "upper_threshold": c,
        }

    def find_quartiles(
        self, data: list[float]
    ) -> typing.Tuple[float, float, float]:
        data = sorted(data)
        q1, q3 = numpy.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        return lower_bound, iqr, upper_bound

    def estimate_ranking_distribution(self) -> list[float]:
        rng = numpy.random.default_rng(self.random_seed)
        # Generate ideal ranking with ranking starting from 1
        ideal_rank = numpy.arange(self.ncols) + 1
        vstacked_list: list[float] = []
        for _ in range(self.n_batches):
            rnd_order = rng.permuted(
                numpy.tile(ideal_rank, self.n_samples).reshape(
                    self.n_samples, ideal_rank.size
                ),
                axis=1,
            )
            # Compute mean over rows
            result = numpy.mean(numpy.abs(ideal_rank - rnd_order), axis=0)
            vstacked_list.extend(result)
        return vstacked_list

    def plot_ranking_distribution(self) -> None:
        # TODO
        return None
