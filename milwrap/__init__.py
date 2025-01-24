"""The milwrap package - multiple instane meta-learner that can use any supervised-learning algorithms.

This package provides implementations of multiple instance learning algorithms
that can wrap around any supervised learning algorithm.
"""

from milwrap.countbase import MilCountBasedMultiClassLearner

__all__ = ["MilCountBasedMultiClassLearner"]
