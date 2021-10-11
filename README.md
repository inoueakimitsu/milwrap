# milwrap

[![Build Status](https://app.travis-ci.com/inoueakimitsu/milwrap.svg?branch=main)](https://app.travis-ci.com/inoueakimitsu/milwrap)
<a href="https://github.com/inoueakimitsu/milwrap/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/inoueakimitsu/milwrap"></a> 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inoueakimitsu/milwrap/blob/master/introduction.ipynb)

Python package for multiple instance learning (MIL).
This wraps single instance learning algorithms so that they can be fitted to data for MIL.

## Features

- support count-based multiple instance assumptions (see [wikipedia](https://en.wikipedia.org/wiki/Multiple_instance_learning#:~:text=Presence-%2C%20threshold-%2C%20and%20count-based%20assumptions%5Bedit%5D))
- support multi-class setting
- support scikit-learn algorithms (such as `RandomForestClassifier`, `SVC`, `LogisticRegression`)

## Installation

```bash
pip install milwrap
```

## Usage

```python
# Prepare single-instance supervised-learning algorithm
# Note: only supports models with predict_proba() method.
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

# Wrap it with MilCountBasedMultiClassLearner
from milwrap import MilCountBasedMultiClassLearner 
mil_learner = MilCountBasedMultiClassLearner(clf)

# Prepare follwing dataset
#
# - bags ... list of np.ndarray
#            (num_instance_in_the_bag * num_features)
# - lower_threshold ... np.ndarray (num_bags * num_classes)
# - upper_threshold ... np.ndarray (num_bags * num_classes)
#
# bags[i_bag] contains not less than lower_thrshold[i_bag, i_class]
# i_class instances.

# run multiple instance learning
clf_mil, y_mil = learner.fit(
    bags,
    lower_threshold,
    upper_threshold,
    n_classes,
    max_iter=10)

# after multiple instance learning,
# you can predict instance class
clf_mil.predict([instance_feature])
```

See `tests/test_countbased.py` for an example of a fully working test data generation process.

## License

milwrap is available under the MIT License.
