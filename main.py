# Import necessary 3rd party libraries
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification

# Local imports
from shapfire import ShapFire

n_splits = 2
n_repeats= 2
estimator_class = LGBMClassifier
scoring = "roc_auc"
DEFAULT_RANDOM_SEED: int = 123
np.random.seed(DEFAULT_RANDOM_SEED)

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=6,
    n_redundant=4,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=False,
)
X = pd.DataFrame(data=X, columns=[f"feature{i}" for i in range(X.shape[1])])

# Instantiate ShapFire method object
estimator_class=LGBMClassifier
estimator_params={"objective": "binary"}
scoring="roc_auc"
n_splits = 2
n_repeats= 5
sf = ShapFire(
  estimator_class=estimator_class,
  scoring=scoring,
  estimator_params=estimator_params,
  n_splits=n_splits,
  n_repeats=n_repeats,
)

# Perform feature importance ranking and selection
_ = sf.fit(X=X, y=y)
