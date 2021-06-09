struct LsqFitResult{P, R, J, W <: AbstractArray}
    param::P
    resid::R
    jacobian::J
    converged::Bool
    wt::W
end


StatsBase.coef(lfr::LsqFitResult) = lfr.param
StatsBase.dof(lfr::LsqFitResult) = nobs(lfr) - length(coef(lfr))
StatsBase.nobs(lfr::LsqFitResult) = length(lfr.resid)
StatsBase.rss(lfr::LsqFitResult) = sum(abs2, lfr.resid)
StatsBase.weights(lfr::LsqFitResult) = lfr.wt
StatsBase.residuals(lfr::LsqFitResult) = lfr.resid
mse(lfr::LsqFitResult) = rss(lfr)/dof(lfr)

function check_data_health(xdata, ydata)
    if any(ismissing, xdata) || any(ismissing, ydata)
        error("Data contains `missing` values and a fit cannot be performed")
    end
    if any(isinf, xdata) || any(isinf, ydata) || any(isnan, xdata) || any(isnan, ydata)
        error("Data contains `Inf` or `NaN` values and a fit cannot be performed")
    end
end

# provide a method for those who have their own Jacobian function
function lmfit(f, g, p0::AbstractArray, wt::AbstractArray; kwargs...)
    r = f(p0)
    R = OnceDifferentiable(f, g, p0, copy(r); inplace=false)
    lmfit(R, p0, wt; kwargs...)
end

#for inplace f and inplace g
function lmfit(f!, g!, p0::AbstractArray, wt::AbstractArray, r::AbstractArray; kwargs...)
    R = OnceDifferentiable(f!, g!, p0, copy(r); inplace = true)
    lmfit(R, p0, wt; kwargs...)
end

#for inplace f only
function lmfit(f, p0::AbstractArray, wt::AbstractArray, r::AbstractArray; autodiff = :finite, kwargs...)
    R = OnceDifferentiable(f, p0, copy(r); inplace = true, autodiff = autodiff)
    lmfit(R, p0, wt; kwargs...)
end

function lmfit(f, p0::AbstractArray, wt::AbstractArray; autodiff = :finite, kwargs...)
    # this is a convenience function for the curve_fit() methods
    # which assume f(p) is the cost functionj i.e. the residual of a
    # model where
    #   model(xpts, params...) = ydata + error (noise)

    # this minimizes f(p) using a least squares sum of squared error:
    #   rss = sum(f(p)^2)
    #
    # returns p, f(p), g(p) where
    #   p    : best fit parameters
    #   f(p) : function evaluated at best fit p, (weighted) residuals
    #   g(p) : estimated Jacobian at p (Jacobian with respect to p)

    # construct Jacobian function, which uses finite difference method
    r = f(p0)
    autodiff = autodiff == :forwarddiff ? :forward : autodiff
    R = OnceDifferentiable(f, p0, copy(r); inplace = false, autodiff = autodiff)
    lmfit(R, p0, wt; kwargs...)
end

function lmfit(R::OnceDifferentiable, p0::AbstractArray, wt::AbstractArray; autodiff = :finite, kwargs...)
    results = levenberg_marquardt(R, p0; kwargs...)
    p = minimizer(results)
    return LsqFitResult(p, value!(R, p), jacobian!(R, p), converged(results), wt)
end

"""
    curve_fit(model, xdata, ydata, p0) -> fit
    curve_fit(model, xdata, ydata, wt, p0) -> fit

Fit data to a non-linear `model`. `p0` is an initial model parameter guess (see Example),
and `wt` is an optional array of weights.
The return object is a composite type (`LsqFitResult`), with some interesting values:

* `fit.resid` : residuals = vector of residuals
* `fit.jacobian` : estimated Jacobian at solution

additionally, it is possible to quiry the degrees of freedom with

* `dof(fit)`
* `coef(fit)`

## Example
```julia
# a two-parameter exponential model
# x: array of independent variables
# p: array of model parameters
model(x, p) = p[1]*exp.(-x.*p[2])

# some example data
# xdata: independent variables
# ydata: dependent variable
xdata = range(0, stop=10, length=20)
ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))
p0 = [0.5, 0.5]

fit = curve_fit(model, xdata, ydata, p0)
```
"""
function curve_fit end

function curve_fit(model, xdata::AbstractArray, ydata::AbstractArray, p0::AbstractArray; inplace = false, kwargs...)
    check_data_health(xdata, ydata)
    # construct the cost function
    T = eltype(ydata)

    if inplace
        f! = (F,p)  -> (model(F,xdata,p); @. F = F - ydata)
        lmfit(f!, p0, T[], ydata; kwargs...)
    else
        f = (p) -> model(xdata, p) - ydata
        lmfit(f, p0, T[]; kwargs...)
    end
end

function curve_fit(model, jacobian_model,
            xdata::AbstractArray, ydata::AbstractArray, p0::AbstractArray; inplace = false, kwargs...)
    check_data_health(xdata, ydata)

    T = eltype(ydata)

    if inplace
        f! = (F,p) -> (model(F,xdata,p); @. F = F - ydata)
        g! = (G,p)  -> jacobian_model(G, xdata, p)
        lmfit(f!, g!, p0, T[], copy(ydata); kwargs...)
    else
        f = (p) -> model(xdata, p) - ydata
        g = (p) -> jacobian_model(xdata, p)
        lmfit(f, g, p0, T[]; kwargs...)
    end
end

function curve_fit(model, xdata::AbstractArray, ydata::AbstractArray, wt::AbstractArray{T}, p0::AbstractArray; inplace = false, kwargs...) where T
    check_data_health(xdata, ydata)
    # construct a weighted cost function, with a vector weight for each ydata
    # for example, this might be wt = 1/sigma where sigma is some error term
    u = sqrt.(wt) # to be consistant with the matrix form

    if inplace
        f! = (F,p) -> (model(F,xdata,p); @. F = u*(F - ydata))
        lmfit(f!, p0, wt, ydata; kwargs...)
    else
        f = (p)  -> u .* ( model(xdata, p) - ydata )
        lmfit(f,p0,wt; kwargs...)
    end
end

function curve_fit(model, jacobian_model,
            xdata::AbstractArray, ydata::AbstractArray, wt::AbstractArray{T}, p0; inplace = false, kwargs...) where T
    check_data_health(xdata, ydata)
    u = sqrt.(wt) # to be consistant with the matrix form

    if inplace
        f! = (F,p) -> (model(F,xdata,p); @. F = u*(F - ydata))
        g! = (G,p) -> (jacobian_model(G, xdata, p); @. G = u*G )
        lmfit(f!, g!, p0, wt, ydata; kwargs...)
    else
        f = (p) -> u .* ( model(xdata, p) - ydata )
        g = (p) -> u .* ( jacobian_model(xdata, p) )
        lmfit(f, g, p0, wt; kwargs...)
    end
end

function curve_fit(model, xdata::AbstractArray, ydata::AbstractArray, wt::AbstractArray{T,2}, p0; kwargs...) where T
    check_data_health(xdata, ydata)

    # as before, construct a weighted cost function with where this
    # method uses a matrix weight.
    # for example: an inverse_covariance matrix

    # Cholesky is effectively a sqrt of a matrix, which is what we want
    # to minimize in the least-squares of levenberg_marquardt()
    # This requires the matrix to be positive definite
    u = cholesky(wt).U

    f(p) = u * ( model(xdata, p) - ydata )
    lmfit(f,p0,wt; kwargs...)
end

function curve_fit(model, jacobian_model,
            xdata::AbstractArray, ydata::AbstractArray, wt::AbstractArray{T,2}, p0; kwargs...) where T
    check_data_health(xdata, ydata)

    u = cholesky(wt).U

    f(p) = u * ( model(xdata, p) - ydata )
    g(p) = u * ( jacobian_model(xdata, p) )
    lmfit(f, g, p0, wt; kwargs...)
end

function estimate_covar(fit::LsqFitResult)
    # computes covariance matrix of fit parameters
    J = fit.jacobian

    if isempty(fit.wt)
        r = fit.resid

        # compute the covariance matrix from the QR decomposition
        Q, R = qr(J)
        Rinv = inv(R)
        covar = Rinv*Rinv'*mse(fit)
    else
        covar = inv(J'*J)
    end

    return covar
end

function StatsBase.stderror(fit::LsqFitResult; rtol::Real=NaN, atol::Real=0)
    # computes standard error of estimates from
    #   fit   : a LsqFitResult from a curve_fit()
    #   atol  : absolute tolerance for approximate comparisson to 0.0 in negativity check
    #   rtol  : relative tolerance for approximate comparisson to 0.0 in negativity check
    covar = estimate_covar(fit)
    # then the standard errors are given by the sqrt of the diagonal
    vars = diag(covar)
    vratio = minimum(vars)/maximum(vars)
    if !isapprox(vratio, 0.0, atol=atol, rtol=isnan(rtol) ? Base.rtoldefault(vratio, 0.0, 0) : rtol) && vratio < 0.0
        error("Covariance matrix is negative for atol=$atol and rtol=$rtol")
    end
    return sqrt.(abs.(vars))
end

function margin_error(fit::LsqFitResult, alpha=0.05; rtol::Real=NaN, atol::Real=0)
    # computes margin of error at alpha significance level from
    #   fit   : a LsqFitResult from a curve_fit()
    #   alpha : significance level, e.g. alpha=0.05 for 95% confidence
    #   atol  : absolute tolerance for approximate comparisson to 0.0 in negativity check
    #   rtol  : relative tolerance for approximate comparisson to 0.0 in negativity check
    std_errors = stderror(fit; rtol=rtol, atol=atol)
    dist = TDist(dof(fit))
    critical_values = quantile(dist, 1 - alpha/2)
    # scale standard errors by quantile of the student-t distribution (critical values)
    return std_errors * critical_values
end

function confidence_interval(fit::LsqFitResult, alpha=0.05; rtol::Real=NaN, atol::Real=0)
    # computes confidence intervals at alpha significance level from
    #   fit   : a LsqFitResult from a curve_fit()
    #   alpha : significance level, e.g. alpha=0.05 for 95% confidence
    #   atol  : absolute tolerance for approximate comparisson to 0.0 in negativity check
    #   rtol  : relative tolerance for approximate comparisson to 0.0 in negativity check
    std_errors = stderror(fit; rtol=rtol, atol=atol)
    margin_of_errors = margin_error(fit, alpha; rtol=rtol, atol=atol)
    confidence_intervals = collect(zip(coef(fit) - margin_of_errors, coef(fit) + margin_of_errors))
end

"""
   tss(data)
    
Compute the total sum of squares of the input `data`.
The output is proportional to the variance of the data.
"""
function tss(data)
    ȳ = sum(data)/length(data)
    SS_tot = sum(y->(y-ȳ)^2, data)
end

"""
    r2(fit::LsqFitResult, ydata)
    r²(fit::LsqFitResult, ydata)

Calculate the "most general definition of the coefficient of determination"
as prescribed in "https://en.wikipedia.org/wiki/Coefficient_of_determination#Definitions".

# Examples
```julia-repl
julia> using LsqFit

julia> xs = LinRange(0, 1, 500)
500-element LinRange{Float64}:

julia> freq = 5
5

julia> ys =  [cos(2π*freq*x) + 1*rand() + 2 for x in xs]
500-element Vector{Float64}:

julia> model(x, p) = @. cos(2π*p[1]*x)+p[2]
model (generic function with 1 method)

julia> p0 = [5, 0.5]
2-element Vector{Float64}:

julia> myfit = curve_fit(model, xs, ys, p0)
LsqFit.LsqFitResult{Vector{Float64}, Vector{Float64}, Matrix{Float64}, Vector{Float64}}...

julia> r2(myfit, ys)
0.8553690031840835
"""
function StatsBase.r2(lfr::LsqFitResult, ydata::AbstractArray)
    return 1-rss(lfr)/tss(ydata)
end

const r² = r2
@deprecate standard_errors(args...; kwargs...) stderror(args...; kwargs...)
@deprecate estimate_errors(fit::LsqFitResult, confidence=0.95; rtol::Real=NaN, atol::Real=0) margin_error(fit, 1-confidence; rtol=rtol, atol=atol)
