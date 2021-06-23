@recipe function f(model::Function, fit::LsqFitResult; significance=0.05, purpose=:neither)
    @series begin
        seriestype --> :line
        label --> "Fit"
        x->model(x, fit.param)
    end
    if purpose in (:confidence, :both)
        @series begin
            seriestype := :line
            seriescolor := :black
            linestyle := :dash
            label := ["Confidence interval" nothing]

            [
             x -> model(x, fit.param) - margin_error(model, x, fit, significance),
             x -> model(x, fit.param) + margin_error(model, x, fit, significance)
            ]
        end
    end
    if purpose in (:prediction, :both)
        @series begin
            seriestype := :line
            seriescolor := :black
            linestyle := :dot
            label := ["Prediction interval" nothing]

            [
             x -> model(x, fit.param) - margin_error(model, x, fit, significance; purpose=:prediction)
             x -> model(x, fit.param) + margin_error(model, x, fit, significance; purpose=:prediction)
            ]
        end
    end
end


"""
```julia
margin_error(model, x, fit, significance; purpose)
```
Find the width at `x` of the confidence or prediction interval.
"""
function margin_error(model::Function, x, fit::LsqFitResult, alpha=0.05; purpose=:confidence)
    g = p -> ForwardDiff.gradient(p -> model(x, p), fit.param)
    c = g(fit.param)' * estimate_covar(fit) * g(fit.param)
    if purpose === :prediction
        c = c + 1
    end
    dist = TDist(dof(fit))
    critical_values = quantile(dist, 1 - alpha/2)
    return sqrt(c*rss(fit)/dof(fit))*critical_values
end
