
# Getting Started

First, import the package.

```julia
julia> using LsqFit
```

Define a two-parameter exponential decay model, where ``t`` is a one-element independent variable, ``p_1`` and ``p_2`` are __float__ parameters.

The model function is:

```math
m(t, \boldsymbol{p}) = p_1 \exp(-p_2 t)
```

```julia
julia> # t: array of independent variable
julia> # p: array of model parameters
julia> model(t, p) = p[1] * exp.(-p[2] * t)
```

For illustration purpose, we generate some fake data.

```julia
julia> # tdata: data of independent variable
julia> # ydata: data of dependent variable
julia> tdata = range(0, stop=10, length=20)
julia> ydata = model(tdata, [1.0 2.0]) + 0.01*randn(length(tdata))
```

Before fitting the data, we also need a initial value of parameters for `curve_fit()`.

```julia
julia> p0 = [0.5, 0.5]
```

Run `curve_fit()` to fit the data and get the estimated parameters.

```julia
julia> fit = curve_fit(model, tdata, ydata, p0)
julia> param = fit.param
2-element Array{Float64,1}:
 1.01105
 2.0735
```

`LsqFit.jl` also provides functions to examinep0 = [0.5, 0.5] the goodness of fit. `estimate_covar(fit)` computes the estimated covariance matrix.

```Julia
julia> cov = estimate_covar(fit)
2×2 Array{Float64,2}:
 0.000116545  0.000174633
 0.000174633  0.00258261
```

`standard_error(fit)` returns the standard error of each parameter.

```Julia
julia> se = standard_error(fit)
2-element Array{Float64,1}:
 0.0107956
 0.0508193
```

To get the confidence interval at 10% significance level, run `confidence_interval(fit, alpha)`, which essentially computes `the estimate parameter value` ± (`standard error` * `critical value from t-distribution`).

```Julia
julia> confidence_interval = confidence_interval(fit, 0.1)
2-element Array{Tuple{Float64,Float64},1}:
 (0.992333, 1.02977)
 (1.98537, 2.16162)
```

For more details of `LsqFit.jl`, check [Tutorial](../tutorial/) and [API References](../api/) section.
