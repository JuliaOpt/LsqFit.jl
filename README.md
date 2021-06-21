LsqFit.jl
===========

The LsqFit package is a small library that provides basic least-squares fitting in pure Julia under an MIT license. The basic functionality was originaly in [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), before being separated into this library.  At this time, `LsqFit` only utilizes the Levenberg-Marquardt algorithm for non-linear fitting.

[![Build Status](https://travis-ci.org/JuliaNLSolvers/LsqFit.jl.svg)](https://travis-ci.org/JuliaNLSolvers/LsqFit.jl)
[![latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://julianlsolvers.github.io/LsqFit.jl/latest/)

Basic Usage
-----------

There are top-level methods `curve_fit()` and `estimate_errors()` that are useful for fitting data to non-linear models. See the following example. Let's first define the model function:
```julia
using LsqFit

# a two-parameter exponential model
# x: array of independent variables
# p: array of model parameters
# model(x, p) will accept the full data set as the first argument `x`.
# This means that we need to write our model function so it applies
# the model to the full dataset. We use `@.` to apply the calculations
# across all rows.
@. model(x, p) = p[1]*exp(-x*p[2])
```
The function applies the per observation function `p[1]*exp(-x[i]*p[2])` to the full dataset in `x`, with `i` denoting an observation row. We simulate some data and chose our "true" parameters.
```julia
# some example data
# xdata: independent variables
# ydata: dependent variable
xdata = range(0, stop=10, length=20)
ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))
p0 = [0.5, 0.5]
```
Now, we're ready to fit the model.
```julia
fit = curve_fit(model, xdata, ydata, p0)
# fit is a composite type (LsqFitResult), with some interesting values:
#	dof(fit): degrees of freedom
#	coef(fit): best fit parameters
#	fit.resid: residuals = vector of residuals
#	fit.jacobian: estimated Jacobian at solution
lb = [1.1, -0.5]
ub = [1.9, Inf]
p0_bounds = [1.2, 1.2] # we have to start inside the bounds
# Optional upper and/or lower bounds on the free parameters can be passed as an argument.
# Bounded and unbouded variables can be mixed by setting `-Inf` if no lower bounds
# is to be enforced for that variable and similarly for `+Inf`
fit_bounds = curve_fit(model, xdata, ydata, p0_bounds, lower=lb, upper=ub)

# We can estimate errors on the fit parameters,
# to get standard error of each parameter:
sigma = stderror(fit)
# to get margin of error and confidence interval of each parameter at 5% significance level:
margin_of_error = margin_error(fit, 0.05)
confidence_inter = confidence_interval(fit, 0.05)

# The finite difference method is used above to approximate the Jacobian.
# Alternatively, a function which calculates it exactly can be supplied instead.
function jacobian_model(x,p)
    J = Array{Float64}(undef, length(x), length(p))
    @. J[:,1] = exp(-x*p[2])     #dmodel/dp[1]
    @. @views J[:,2] = -x*p[1]*J[:,1] #dmodel/dp[2], thanks to @views we don't allocate memory for the J[:,1] slice
    J
end
fit = curve_fit(model, jacobian_model, xdata, ydata, p0)
```

Multivariate regression
-----------------------
There's nothing inherently different if there are more than one variable entering the problem. We just need to specify the columns appropriately in our model specification:
```julia
@. multimodel(x, p) = p[1]*exp(-x[:, 1]*p[2]+x[:, 2]*p[3])
```
Evaluating the Jacobian and using automatic differentiation
-------------------------
The default is to calculate the Jacobian using a central finite differences scheme if no Jacobian function is provided. The defaul is to use central differences because it can be more accurate than forward finite differences, but at the expense of computational cost. It is possible to switch to forward finite differences, like MINPACK uses for example, by specifying `autodiff=:finiteforward`:
```julia
fit = curve_fit(model, xdata, ydata, p0; autodiff=:finiteforward)
```
It is also possible to use forward mode automatic differentiation as implemented in ForwardDiff.jl by using the `autodiff=:forwarddiff` keyword.
```julia
fit = curve_fit(model, xdata, ydata, p0; autodiff=:forwarddiff)
```
Here, you have to be careful not to manually restrict any types in your code to, say, `Float64`, because ForwardDiff.jl works by passing a special number type through your functions, to auto*magically* calculate the value and gradient with one evaluation.

Inplace model and jacobian
-------------------------
It is possible to either use an inplace model, or an inplace model *and* an inplace jacobian. It might be pertinent to use this feature when `curve_fit` is slow, or consumes a lot of memory
```julia
model_inplace(F, x, p) = (@. F = p[1] * exp(-x * p[2]))

function jacobian_inplace(J::Array{Float64,2},x,p)
        @. J[:,1] = exp(-x*p[2])     
        @. @views J[:,2] = -x*p[1]*J[:,1]
    end
fit = curve_fit(model_inplace, jacobian_inplace, xdata, ydata, p0; inplace = true)
```

Geodesic acceleration
---------------------
This package implements optional geodesic acceleration, as outlined by [this paper](https://arxiv.org/pdf/1010.1449.pdf). To enable it, one needs to specify the function computing the *[directional second derivative](https://math.stackexchange.com/questions/2342410/why-is-mathbfdt-h-mathbfd-the-second-directional-derivative)* of the function that is fitted, as the `avv!` parameter. It is also preferable to set `lambda` and `min_step_quality`to `0`:
```julia
curve_fit(model, xdata, ydata, p0; avv! = Avv!,lambda=0, min_step_quality = 0)
```
`Avv!` must have the following form:
* `p` is the array of parameters
* `v`is the direction in which the direction is taken
* `dir_deriv` is the output vector (the function is necessarily inplace)
```julia
function Avv!(dir_deriv,p,v)
        v1 = v[1]
        v2 = v[2]
        for i=1:length(xdata)
            #compute all the elements of the Hessian matrix
            h11 = 0
            h12 = (-xdata[i] * exp(-xdata[i] * p[2]))
            #h21 = h12
            h22 = (xdata[i]^2 * p[1] * exp(-xdata[i] * p[2]))

            # manually compute v'Hv. This whole process might seem cumbersome, but
            # allocating temporary matrices quickly becomes REALLY expensive and might even
            # render the use of geodesic acceleration terribly inefficient  
            dir_deriv[i] = h11*v1^2 + 2*h12*v1*v2 + h22*v2^2

        end
end
```
Typically, if the model to fit outputs `[y_1(x),y_2(x),...,y_m(x)]`, and that the input data is `xdata` then `Avv!`should output an array of size `m`, where each element is `v'*H_i(xdata,p)*v`, where `H_i`is the Hessian matrix of the output `y_i`with respect to the parameter vector `p`.

Depending on the size of the dataset, the complexity of the model and the desired tolerance in the fit result, it may be worthwhile to use automatic differentiation (e.g. via `Zygote.jl` or `ForwardDiff.jl`) to determine the directional derivative. Although this is potentially less efficient than calculating the directional derivative manually, this additional information will generally lead to more accurate results.

An example of such an implementation is given by:
```julia
using LinearAlgebra, Zygote

function Avv!(dir_deriv,p,v)
    for i=1:length(xdata)
        dir_deriv[i] = transpose(v) * Zygote.hessian(z->model(xdata[i],z),p) * v
    end
end
```

Existing Functionality
----------------------

`fit = curve_fit(model, [jacobian], x, y, [w,] p0; kwargs...)`:

* `model`: function that takes two arguments (x, params)
* `jacobian`: (optional) function that returns the Jacobian matrix of `model`
* `x`: the independent variable
* `y`: the dependent variable that constrains `model`
* `w`: (optional) weight applied to the residual; can be a vector (of `length(x)` size or empty) or matrix (inverse covariance matrix)
* `p0`: initial guess of the model parameters
* `kwargs`: tuning parameters for fitting, passed to `levenberg_marquardt`, such as `maxIter`, `show_trace` or `lower` and `upper` bounds
* `fit`: composite type of results (`LsqFitResult`)


This performs a fit using a non-linear iteration to minimize the (weighted) residual between the model and the dependent variable data (`y`). The weight (`w`) can be neglected (as per the example) to perform an unweighted fit. An unweighted fit is the numerical equivalent of `w=1` for each point  (although unweighted error estimates are handled differently from weighted error estimates even when the weights are uniform).

----

`sigma = stderror(fit; atol, rtol)`:

* `fit`: result of curve_fit (a `LsqFitResult` type)
* `atol`: absolute tolerance for negativity check
* `rtol`: relative tolerance for negativity check

This returns the error or uncertainty of each parameter fit to the model and already scaled by the associated degrees of freedom.  Please note, this is a LOCAL quantity calculated from the jacobian of the model evaluated at the best fit point and NOT the result of a parameter exploration.

If no weights are provided for the fits, the variance is estimated from the mean squared error of the fits. If weights are provided, the weights are assumed to be the inverse of the variances or of the covariance matrix, and errors are estimated based on these and the jacobian, assuming a linearization of the model around the minimum squared error point.

`margin_of_error = margin_error(fit, alpha=0.05; atol, rtol)`:

* `fit`: result of curve_fit (a `LsqFitResult` type)
* `alpha`: significance level
* `atol`: absolute tolerance for negativity check
* `rtol`: relative tolerance for negativity check

This returns the product of standard error and critical value of each parameter at `alpha` significance level.

`confidence_interval = confidence_interval(fit, alpha=0.05; atol, rtol)`:

* `fit`: result of curve_fit (a `LsqFitResult` type)
* `alpha`: significance level
* `atol`: absolute tolerance for negativity check
* `rtol`: relative tolerance for negativity check

This returns confidence interval of each parameter at `alpha` significance level.

----

`covar = estimate_covar(fit)`:

* `fit`: result of curve_fit (a `LsqFitResult` type)
* `covar`: parameter covariance matrix calculated from the jacobian of the model at the fit point, using the weights (if specified) as the inverse covariance of observations

This returns the parameter covariance matrix evaluted at the best fit point.
