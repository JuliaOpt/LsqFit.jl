# Tutorial

## Introduction to Nonlinear Regression

Assume that, for the $i$th observation, the relationship between independent variable $\mathbf{x_i}=\begin{bmatrix} x_{1i},\, x_{2i},\, \ldots\, x_{pi}\ \end{bmatrix}'$ and dependent variable $Y_i$ follows:

```math
Y_i = m(\mathbf{x_i}, \boldsymbol{\gamma}) + \epsilon_i
```

where $m$ is a non-linear model function depends on the independent variable $\mathbf{x_i}$ and the parameter vector $\boldsymbol{\gamma}$. In order to find the parameter $\boldsymbol{\gamma}$ that "best" fit our data, we choose the parameter ${\boldsymbol{\gamma}}$ which minimizes the sum of squared residuals from our data, i.e. solves the problem:

```math
\underset{\boldsymbol{\gamma}}{\mathrm{min}} \quad s(\boldsymbol{\gamma})= \sum_{i=1}^{n} [m(\mathbf{x_i}, \boldsymbol{\gamma}) - y_i]^2
```

Given that the function $m$ is non-linear, there's no analytical solution for the best $\boldsymbol{\gamma}$. We have to use computational tools, which is `LsqFit.jl` in this tutorial, to find the least squares solution.

One example of non-linear model is the exponential model, which takes a one-element predictor variable $t$. The model function is:

```math
m(t, \boldsymbol{\gamma}) = \gamma_1 \exp(\gamma_2 t)
```

and the model becomes:

```math
Y_i = \gamma_1 \exp(\gamma_2 t_i) + \epsilon_i
```

To fit data using `LsqFit.jl`, pass the defined model function (`m`), data (`tdata` and `ydata`) and the initial parameter value (`p0`) to `curve_fit()`. For now, `LsqFit.jl` only supports the Levenberg Marquardt algorithm.

```julia
julia> # t: array of independent variables
julia> # p: array of model parameters
julia> m(t, p) = p[1] * exp.(p[2] * t)
julia> p0 = [0.5, 0.5]
julia> fit = curve_fit(m, tdata, ydata, p0)
```

It will return a composite type `LsqFitResult`, with some interesting values:

*	`dof(fit)`: degrees of freedom
*	`coef(fit)`: best fit parameters
*	`fit.resid`: vector of residuals
*	`fit.jacobian`: estimated Jacobian at the solution

## Jacobian Calculation

The Jacobian $J_f(\mathbf{x})$ of a vector function $f(\mathbf{x}): \mathbb{R}_m \to \mathbb{R}_n$ is deﬁned as the matrix with elements:

```math
[J_f(\mathbf{x})]_{ij} = \frac{\partial f_i(\mathbf{x})}{\partial x_j}
```
The matrix is therefore:

```math
J_f(\mathbf{x}) = \begin{bmatrix}
                \frac{\partial f_1}{\partial x_1}&\frac{\partial f_1}{\partial x_2}&\dots&\frac{\partial f_1}{\partial x_m}\\
                \frac{\partial f_2}{\partial x_1}&\frac{\partial f_2}{\partial x_2}&\dots&\frac{\partial f_2}{\partial x_m}\\
                \vdots&\vdots&\ddots&\vdots\\
                \frac{\partial f_n}{\partial x_1}&\frac{\partial f_n}{\partial x_2}&\dots&\frac{\partial f_n}{\partial x_m}\\
                \end{bmatrix}
```

The Jacobian of the exponential model function with respect to $\boldsymbol{\gamma}$ is:

```math
J_m(t, \boldsymbol{\gamma}) = \begin{bmatrix}
            \frac{\partial m}{\partial \gamma_1} &
            \frac{\partial m}{\partial \gamma_2} \\
            \end{bmatrix}
          = \begin{bmatrix}
            \exp(\gamma_2 t) &
            t \gamma_1 \exp(\gamma_2 t) \\
            \end{bmatrix}
```

By default, the finite differences is used (see [NLSolversBase.jl](https://github.com/JuliaNLSolvers/NLSolversBase.jl) for more information), is used to approximate the Jacobian for the data fitting algorithm and covariance computation. Alternatively, a function which calculates the Jacobian can be supplied to `curve_fit()` for faster and/or more accurate results.

```Julia
function j_m(t,p)
    J = Array{Float64}(length(t),length(p))
    J[:,1] = exp.(p[2] .* t)       #df/dp[1]
    J[:,2] = t .* p[1] .* J[:,1]   #df/dp[2]
    J
end

fit = curve_fit(m, j_m, tdata, ydata, p0)
```

## Linear Approximation

The non-linear function $m$ can be approximated as a linear function by Talor expansion:

```math
m(\mathbf{x_i}, \boldsymbol{\gamma}+\boldsymbol{h}) \approx m(\mathbf{x_i}, \boldsymbol{\gamma}) +  \nabla m(\mathbf{x_i}, \boldsymbol{\gamma})'\boldsymbol{h}
```

where $\boldsymbol{\gamma}$ is a fixed vector, $\boldsymbol{h}$ is a very small-valued vector and $\nabla m(\mathbf{x_i}, \boldsymbol{\gamma})$ is the gradient at $\mathbf{x_i}$.

Consider the residual vector functon $r({\boldsymbol{\gamma}})=\begin{bmatrix}
                          r_1({\boldsymbol{\gamma}}) \\
                          r_2({\boldsymbol{\gamma}}) \\
                          \vdots\\
                          r_n({\boldsymbol{\gamma}})
                          \end{bmatrix}$
with entries:

```math
r_i({\boldsymbol{\gamma}}) = m(\mathbf{x_i}, {\boldsymbol{\gamma}}) - Y_i
```

Each entry's linear approximation can hence be written as:

```math
\begin{align}
r_i({\boldsymbol{\gamma}}+\boldsymbol{h}) &= m(\mathbf{x_i}, \boldsymbol{\gamma}+\boldsymbol{h}) - Y_i\\
&\approx m(\mathbf{x_i}, \boldsymbol{\gamma}) + \nabla m(\mathbf{x_i}, \boldsymbol{\gamma})'h - Y_i\\
&= r_i({\boldsymbol{\gamma}}) + \nabla m(\mathbf{x_i}, \boldsymbol{\gamma})'h
\end{align}
```

Since the $i$th row of $J(\boldsymbol{\gamma})$ equals the transpose of the gradient of $m(\mathbf{x_i}, \boldsymbol{\gamma})$, the vector function $r({\boldsymbol{\gamma}}+\boldsymbol{h})$ can be approximated as:

```math
r({\boldsymbol{\gamma}}+\boldsymbol{h}) \approx r({\boldsymbol{\gamma}}) + J(\boldsymbol{\gamma})h
```

which is a linear function on $\boldsymbol{h}$ since ${\boldsymbol{\gamma}}$ is a fixed vector.

## Goodness of Fit

The linear approximation of the non-linear least squares problem leads to the approximation of the covariance matrix of each parameter, from which we can perform regression analysis.

Consider a least squares solution $\boldsymbol{\gamma}^*$, which is a local minimizer of the non-linear problem:

```math
\boldsymbol{\gamma}^* = \underset{\boldsymbol{\gamma}}{\mathrm{arg\,min}} \ \sum_{i=1}^{n} [m(\mathbf{x_i}, \boldsymbol{\gamma}) - y_i]^2
```

Set $\boldsymbol{\gamma}^*$ as the fixed point in linear approximation, $r({\boldsymbol{\gamma^*}}) = r$ and $J(\boldsymbol{\gamma^*}) = J$. A parameter vector near $\boldsymbol{\gamma}^*$ can be expressed as $\boldsymbol{\gamma}=\boldsymbol{\gamma^*} + h$. The local approximation for the least squares problem is:

```math
\underset{\boldsymbol{\gamma}}{\mathrm{min}} \quad s(\boldsymbol{\gamma})=s(\boldsymbol{\gamma}^*+\boldsymbol{h}) \approx [Jh + r]'[Jh + r]
```

which is essentially the linear least squares problem:

```math
\underset{\boldsymbol{\beta}}{\mathrm{min}} \quad [X\beta-Y]'[X\beta-Y]
```

where $X=J$, $\beta=\boldsymbol{h}$ and $Y=-r({\boldsymbol{\gamma}})$. Solve the equation where the partial derivatives equal to $0$, the analytical solution is:

```math
\hat{\boldsymbol{h}}=\hat{\boldsymbol{\gamma}}-\boldsymbol{\gamma}^*\approx-[J'J]^{-1}J'r
```

The covariance matrix for the analytical solution is:

```math
\mathbf{Cov}(\hat{\boldsymbol{\gamma}}) = \mathbf{Cov}(\boldsymbol{h}) = [J'J]^{-1}J'\mathbf{E}(rr')J[J'J]^{-1}
```

Note that $r$ is the residual vector at the best fit point $\boldsymbol{\gamma^*}$, with entries $r_i = Y_i - m(\mathbf{x_i}, \boldsymbol{\gamma^*})=\epsilon_i$. $\hat{\boldsymbol{\gamma}}$ is very close to $\boldsymbol{\gamma^*}$ and therefore can be replaced by $\boldsymbol{\gamma^*}$.

```math
\mathbf{Cov}(\boldsymbol{\gamma}^*) \approx \mathbf{Cov}(\hat{\boldsymbol{\gamma}})
```

Assume the errors in each sample are independent, normal distributed with zero mean and same variance, i.e. $\epsilon \sim N(0, \sigma^2I)$, the covariance matrix from the linear approximation is therefore:

```math
\mathbf{Cov}(\boldsymbol{\gamma}^*) = [J'J]^{-1}J'\mathbf{Cov}(\epsilon)J[J'J]^{-1} = \sigma^2[J'J]^{-1}
```

where $\sigma^2$ could be estimated as residual sum of squares devided by degrees of freedom:

```math
\hat{\sigma}^2=\frac{s(\boldsymbol{\gamma}^*)}{n-p}
```
In `LsqFit.jl`, the covariance matrix calculation uses QR decomposition to [be more computationally stable](http://www.seas.ucla.edu/~vandenbe/133A/lectures/ls.pdf), which has the form:

```math
\mathbf{Cov}(\boldsymbol{\gamma}^*) = \hat{\sigma}^2 \mathrm{R}^{-1}(\mathrm{R}^{-1})'
```

`estimate_covar()` computes the covariance matrix of fit:

```Julia
julia> cov = estimate_covar(fit)
2×2 Array{Float64,2}:
 0.000116545  0.000174633
 0.000174633  0.00258261
```

The standard error is then the square root of each diagonal elements of the covariance matrix. `standard_error()` returns the standard error of each parameter:

```Julia
julia> se = standard_error(fit)
2-element Array{Float64,1}:
 0.0114802
 0.0520416
```

`margin_error()` computes the product of standard error and the critical value of each parameter at a certain significance level (default is 5%) from t-distribution. The margin of error at 10% significance level can be computed by:

```Julia
julia> margin_of_error = margin_error(fit, 0.1)
2-element Array{Float64,1}:
 0.0199073
 0.0902435
```

`confidence_interval()` returns the confidence interval of each parameter at certain significance level, which is essentially the estimate value ± margin of error. To get the confidence interval at 10% significance level, run:

```Julia
julia> confidence_intervals = confidence_interval(fit, 0.1)
2-element Array{Tuple{Float64,Float64},1}:
 (0.976316, 1.01613)
 (1.91047, 2.09096)
```

`r2()` or `r²()` computes the coefficient of determination. There are multiple definitions, but the one implemented is described as "The most general definition of the coefficient of determination" in [this wikipedia article](https://en.wikipedia.org/wiki/Coefficient_of_determination#Definitions).
```julia
julia> r2(fit, ydata)
0.9985914750863337
```

## Weighted Least Squares

`curve_fit()` also accepts weight parameter (`wt`) to perform Weighted Least Squares and General Least Squares, where the parameter $\boldsymbol{\gamma}^*$ minimizes the weighted residual sum of squares.

Weight parameter (`wt`) is an array or a matrix of weights for each sample. To perform Weighted Least Squares, pass the weight array `[w_1, w_2, ..., w_n]` or the weight matrix `W`:

```math
\mathbf{W} = \begin{bmatrix}
    w_1    & 0      & \cdots & 0\\
    0      & w_2    & \cdots & 0\\
    \vdots & \vdots & \ddots & \vdots\\
    0      & 0    & \cdots & w_n\\
    \end{bmatrix}
```

The weighted least squares problem becomes:

```math
\underset{\boldsymbol{\gamma}}{\mathrm{min}} \quad s(\boldsymbol{\gamma})= \sum_{i=1}^{n} w_i[m(\mathbf{x_i}, \boldsymbol{\gamma}) - Y_i]^2
```

in matrix form:

```math
\underset{\boldsymbol{\gamma}}{\mathrm{min}} \quad s(\boldsymbol{\gamma})= r(\boldsymbol{\gamma})'Wr(\boldsymbol{\gamma})
```

where $r({\boldsymbol{\gamma}})=\begin{bmatrix}
                          r_1({\boldsymbol{\gamma}}) \\
                          r_2({\boldsymbol{\gamma}}) \\
                          \vdots\\
                          r_n({\boldsymbol{\gamma}})
                          \end{bmatrix}$
is a residual vector function with entries:

```math
r_i({\boldsymbol{\gamma}}) = m(\mathbf{x_i}, {\boldsymbol{\gamma}}) - Y_i
```

The algorithm in `LsqFit.jl` will then provide a least squares solution $\boldsymbol{\gamma}^*$.

!!! note
    In `LsqFit.jl`, the residual function passed to `levenberg_marquardt()` is in different format, if the weight is a vector:

    ```julia
    r(p) = sqrt.(wt) .* ( model(xpts, p) - ydata )
    lmfit(r, g, p0, wt; kwargs...)
    ```

    ```math
    r_i({\boldsymbol{\gamma}}) = \sqrt{w_i} \cdot [m(\mathbf{x_i}, {\boldsymbol{\gamma}}) - Y_i]
    ```

    Cholesky decomposition, which is effectively a sqrt of a matrix, will be performed if the weight is a matrix:

    ```julia
    u = chol(wt)
    r(p) = u * ( model(xpts, p) - ydata )
    lmfit(r, p0, wt; kwargs...)
    ```

    ```math
    r_i({\boldsymbol{\gamma}}) = \sqrt{w_i} \cdot [m(\mathbf{x_i}, {\boldsymbol{\gamma}}) - Y_i]
    ```

    The solution will be the same as the least squares problem mentioned in the tutorial.

Set $r({\boldsymbol{\gamma^*}}) = r$ and $J(\boldsymbol{\gamma^*}) = J$, the linear approximation of the weighted least squares problem is then:

```math
\underset{\boldsymbol{\gamma}}{\mathrm{min}} \quad s(\boldsymbol{\gamma}) = s(\boldsymbol{\gamma}^* + \boldsymbol{h}) \approx [J\boldsymbol{h}+r]'W[J\boldsymbol{h}+r]
```

The analytical solution to the linear approximation is:

```math
\hat{\boldsymbol{h}}=\hat{\boldsymbol{\gamma}}-\boldsymbol{\gamma}^*\approx-[J'WJ]^{-1}J'Wr
```

Assume the errors in each sample are independent, normal distributed with zero mean and **different** variances (heteroskedastic error), i.e. $\epsilon \sim N(0, \Sigma)$, where:

```math
\Sigma = \begin{bmatrix}
         \sigma_1^2    & 0      & \cdots & 0\\
         0      & \sigma_2^2    & \cdots & 0\\
         \vdots & \vdots & \ddots & \vdots\\
         0      & 0    & \cdots & \sigma_n^2\\
         \end{bmatrix}
```

We know the error variance and we set the weight as the inverse of the variance (the optimal weight), i.e. $W = \Sigma^{-1}$:

```math
\mathbf{W} =  \begin{bmatrix}
              w_1    & 0      & \cdots & 0\\
              0      & w_2    & \cdots & 0\\
              \vdots & \vdots & \ddots & \vdots\\
              0      & 0    & \cdots & w_n\\
              \end{bmatrix}
           =  \begin{bmatrix}
               \frac{1}{\sigma_1^2}    & 0      & \cdots & 0\\
               0      & \frac{1}{\sigma_2^2}    & \cdots & 0\\
               \vdots & \vdots & \ddots & \vdots\\
               0      & 0    & \cdots & \frac{1}{\sigma_n^2}\\
               \end{bmatrix}
```

The covariance matrix is now:

```math
{Cov}(\boldsymbol{\gamma}^*) \approx  [J'WJ]^{-1}J'W \Sigma W'J[J'W'J]^{-1} = [J'WJ]^{-1}
```


If we only know **the relative ratio of different variances**, i.e. $\epsilon \sim N(0, \sigma^2W^{-1})$, the covariance matrix will be:

```math
\mathbf{Cov}(\boldsymbol{\gamma}^*) = \sigma^2[J'WJ]^{-1}
```

where $\sigma^2$ is estimated. In this case, if we set $W = I$, the result will be the same as the unweighted version. However, `curve_fit()` currently **does not support** this implementation. `curve_fit()` assumes the weight as the inverse of **the error covariance matrix** rather than **the ratio of error covariance matrix**, i.e. the covariance of the estimated parameter is calculated as `covar = inv(J'*fit.wt*J)`.

!!! note
    Passing vector of ones as the weight vector will cause mistakes in covariance estimation.

Pass the vector of `1 ./ var(ε)` or the matrix `inv(covar(ε))` as the weight parameter (`wt`) to the function `curve_fit()`:

```Julia
julia> wt = inv(cov_ε)
julia> fit = curve_fit(m, tdata, ydata, wt, p0)
julia> cov = estimate_covar(fit)
```

!!! note
    If the weight matrix is not a diagonal matrix, General Least Squares will be performed.

## General Least Squares
Assume the errors in each sample are **correlated**, normal distributed with zero mean and **different** variances (heteroskedastic and autocorrelated error), i.e. $\epsilon \sim N(0, \Sigma)$.

Set the weight matrix as the inverse of the error covariance matrix (the optimal weight), i.e. $W = \Sigma^{-1}$, we will get the parameter covariance matrix:

```math
\mathbf{Cov}(\boldsymbol{\gamma}^*) \approx  [J'WJ]^{-1}J'W \Sigma W'J[J'W'J]^{-1} = [J'WJ]^{-1}
```

Pass the matrix `inv(covar(ε))` as the weight parameter (`wt`) to the function `curve_fit()`:

```Julia
julia> wt = 1 ./ yvar
julia> fit = curve_fit(m, tdata, ydata, wt, p0)
julia> cov = estimate_covar(fit)
```

## Estimate the Optimal Weight
In most cases, the variances of errors are unknown. To perform Weighted Least Square, we need estimate the variances of errors first, which is the squared residual of $i$th sample:

```math
\widehat{\mathbf{Var}(\epsilon_i)} = \widehat{\mathbf{E}(\epsilon_i \epsilon_i)} = r_i(\boldsymbol{\gamma}^*)
```

Unweighted fitting (OLS) will return the residuals we need, since the estimator of OLS is unbiased. Then pass the reciprocal of the residuals as the estimated optimal weight to perform Weighted Least Squares:

```Julia
julia> fit_OLS = curve_fit(m, tdata, ydata, p0)
julia> wt = 1 ./ fit_OLS.resid
julia> fit_WLS = curve_fit(m, tdata, ydata, wt, p0)
julia> cov = estimate_covar(fit_WLS)
```

## References
Hansen, P. C., Pereyra, V. and Scherer, G. (2013) Least squares data fitting with applications. Baltimore, Md: Johns Hopkins University Press, p. 147-155.

Kutner, M. H. et al. (2005) Applied Linear statistical models.

Weisberg, S. (2014) Applied linear regression. Fourth edition. Hoboken, NJ: Wiley (Wiley series in probability and statistics).
