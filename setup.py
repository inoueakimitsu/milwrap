"""Setup script for the milwrap package."""

from pathlib import Path

from setuptools import setup

# get key package details from milwrap/__version__.py
about = {}  # type: ignore
here = Path(__file__).resolve().parent
with (here / "milwrap" / "__version__.py").open() as f:
    exec(f.read(), about)  # noqa: S102

# load the README file and use it as the long_description for PyPI
with Path("README.md").open() as f:
    readme = f.read()

setup(
    name=about["__title__"],
    description=about["__description__"],
    long_description=readme,
    long_description_content_type="text/markdown",
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    url=about["__url__"],
    packages=["milwrap"],
    include_package_data=True,
    python_requires=">=3.7.*",
    install_requires=["numpy", "pandas", "scikit-learn"],
    license=about["__license__"],
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="machine learning",
)
