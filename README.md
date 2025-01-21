# `tab_err`

`tab_err` is an implementation of a tabular data error model that disentangles error mechanism and error type.
It generalizes the formalization of missing values, implying that missing values are only one of many possible error type implemented here.
`tab_err` gives the user full control over the error generation process and allows to model realistic errors with complex dependency structures.

The building blocks are `ErrorMechanism`s, `ErrorType`s, and `ErrorModel`s.
`ErrorMechanism` defines where the incorrect cells are and model realistic dependency structures and `ErrorType` describes in which way the value is incorrect.
Together they build a `ErrorModel` that can be used to perturb existing data with realistic errors.

This repository offers (soon) three APIs, low-level, mid-level and high-level.

## Examples

For details and examples please check out our [Getting Started Notebook](https://github.com/calgo-lab/tab_err/blob/main/examples/1-Getting-Started.ipynb).

## Where to get it

The source code is currently hosted on GitHub at:
<https://github.com/calgo-lab/tab_err>

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/tab-err).

```sh
pip install tab-err
```
