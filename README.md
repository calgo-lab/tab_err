# `tab_err`
<h1 align="center">
  Inject Realistic Errors Into Tables
</h1>

<p align="center">
  📊 🔎 ✅
</p>

<p align="center">
  <i>Test how your data pipelines and ML models react to tabular data that contains realistic errors.</i>
</p>

<br>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tab_err)](https://pypi.org/project/tab_err/)
[![mypy](https://github.com/calgo-lab/tab_err/actions/workflows/mypy.yaml/badge.svg?branch=main)](https://github.com/calgo-lab/tab_err/actions/workflows/mypy.yaml)
[![pytest](https://github.com/calgo-lab/tab_err/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/calgo-lab/tab_err/actions/workflows/tests.yaml)
[![Ruff](https://github.com/calgo-lab/tab_err/actions/workflows/ruff.yaml/badge.svg?branch=main)](https://github.com/calgo-lab/tab_err/actions/workflows/ruff.yaml)


`tab_err` injects realistic errors into tabular data such as database tables and DataFrames.
The library is developed and maintained by the [Cognitive Algorithms Lab](https://calgo-lab.de/) at BHT Berlin.

Using error-free tables as input, `tab_err` lets users define an error model that perturbs the table and can be shared as metadata.
Researchers and data practitioners can generate errors in a controlled way, evaluate how their systems behave, and exchange error scenarios reproducibly.

## How it Works

The library's building blocks are `ErrorMechanism`s, `ErrorType`s, and `ErrorModel`s.
- An `ErrorMechanism` describes the error's distribution - that's *where* incorrect cells appear in the table.  We support *erroneous at random* (EAR), *erroneous not at random* (ENAR) and *erroneous completely at random* (ECAR).
- An `ErrorType` describes *how* the value is wrong: a typo, an outlier, a category swap, and so on. Read the documentation for a [full list of supported error types](https://tab-err.readthedocs.io/latest/api/tab_err/error_type/index.html).
- An `ErrorModel` is a set of mechanisms and types to perturb existing data with realistic errors. It is shareable as metadata.

`tab_err` is supported by a `pandas` backend.

## Examples
```python
from sklearn.datasets import load_iris

from tab_err import error_type
from tab_err.api import high_level

df = load_iris(as_frame=True).frame
corrupted_df, error_mask = high_level.create_errors(
    data=df,
    error_rate=0.5,
    error_types_to_exclude=[error_type.MissingValue()],
    seed=42,
)
print("Original:")
print(df.head(2).to_string(index=False))

print("\nCorrupted:")
print(corrupted_df.head(2).to_string(index=False))

print("\nCorrupted cells:", int(error_mask.to_numpy().sum()))
```

Example output:

```text
Original:
 sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
               5.1               3.5                1.4               0.2       0
               4.9               3.0                1.4               0.2       0

Corrupted:
 sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)   target
               5.1              35.0           1.400000         -2.775759 0.420326
               4.9              30.0           1.820326         -3.087558 0.000000

Corrupted cells: 375
```

For a detailed guide and more examples, see our [Getting Started Notebook](https://github.com/calgo-lab/tab_err/blob/main/examples/1-Getting-Started.ipynb) and the [documentation](https://tab-err.readthedocs.io/latest/).

## Where to get it

The source code is hosted on GitHub at <https://github.com/calgo-lab/tab_err>. 
Binary installers for the latest releases are available at the Python Package Index (PyPI) <https://pypi.org/project/tab-err>.

```sh
# with pip
pip install tab-err

# with uv
uv add tab-err
```

## Contributing

To develop `tab_err`, install the `uv` package manager.
Run tests with `uv run pytest`.
Develop on feature branches and open pull requests when you're ready.
Make sure that your changes are tested, documented, and clearly described in the pull request.

## Citation
If you use the error model that's underlying `tab_err` for a scientific publication, we would appreciate your citation.

```
@article{10.1145/3774914,
author = {Jung, Philipp and J\"{a}ger, Sebastian and Chandler, Nicholas and Biessmann, Felix},
title = {Towards Realistic Error Models for Tabular Data},
year = {2025},
issue_date = {December 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {17},
number = {4},
issn = {1936-1955},
url = {https://doi.org/10.1145/3774914},
doi = {10.1145/3774914},
journal = {J. Data and Information Quality},
month = dec,
articleno = {28},
numpages = {27},
keywords = {Tabular data, data quality, data errors, data error generation, error model, realistic error model, error type}
}
```