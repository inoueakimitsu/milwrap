"""Information about the current version of the milwrap package."""

from importlib.metadata import PackageNotFoundError, version

__title__ = "milwrap"
__description__ = "milwrap - multiple instane meta-learner that can use any supervised-learning algorithms."
try:
    __version__ = version(__title__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
__author__ = "Akimitsu Inoue"
__author_email__ = "akimitsu.inoue@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/inoueakimitsu/milwrap"
