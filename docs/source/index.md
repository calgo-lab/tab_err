# `tab_err`

`tab_err` is an implementation of a tabular data error model that disentangles error mechanism and error type.
It generalizes the formalization of missing values, implying that missing values are only one of many possible error type implemented here.
`tab_err` gives the user full control over the error generation process and allows to model realistic errors with complex dependency structures.

The building blocks are [`ErrorMechanism`s](api/tab_err/error_mechanism/index), [`ErrorType`s](api/tab_err/error_type/index), and [`ErrorModel`s](api/tab_err/index).
[`ErrorMechanism`](api/tab_err/error_mechanism/index) defines where the incorrect cells are and model realistic dependency structures and [`ErrorType`](api/tab_err/error_type/index) describes in which way the value is incorrect.
Together they build a [`ErrorModel`](api/tab_err/index) that can be used to perturb existing data with realistic errors.

## Examples

For details and examples please check out our [Getting Started Notebook](https://github.com/calgo-lab/tab_err/blob/main/examples/1-Getting-Started.ipynb).

## API References

```{toctree}
:maxdepth: 4

API References <api/index>
```
