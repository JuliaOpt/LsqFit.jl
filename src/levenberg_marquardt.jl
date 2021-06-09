struct LevenbergMarquardt <: Optimizer end
Base.summary(::LevenbergMarquardt) = "Levenberg-Marquardt"
"""
    `levenberg_marquardt(f, g, initial_x; <keyword arguments>`

Returns the argmin over x of `sum(f(x).^2)` using the Levenberg-Marquardt
algorithm, and an estimate of the Jacobian of `f` at x.

The function `f` should take an input vector of length n and return an output
vector of length m. The function `g` is the Jacobian of f, and should return an m x
n matrix. `initial_x` is an initial guess for the solution.

Implements box constraints as described in Kanzow, Yamashita, Fukushima (2004; J
Comp & Applied Math).

# Keyword arguments
* `x_tol::Real=1e-8`: search tolerance in x
* `g_tol::Real=1e-12`: search tolerance in gradient
* `maxIter::Integer=1000`: maximum number of iterations
* `min_step_quality=1e-3`: for steps below this quality, the trust region is shrinked
* `good_step_quality=0.75`: for steps above this quality, the trust region is expanded
* `lambda::Real=10`: (inverse of) initial trust region radius
* `tau=Inf`: set initial trust region radius using the heuristic : tau*maximum(jacobian(df)'*jacobian(df))
* `lambda_increase=10.0`: `lambda` is multiplied by this factor after step below min quality
* `lambda_decrease=0.1`: `lambda` is multiplied by this factor after good quality steps
* `show_trace::Bool=false`: print a status summary on each iteration if true
* `lower,upper=[]`: bound solution to these limits
"""

# I think a smarter way to do this *might* be to create a type similar to `OnceDifferentiable`
# and the like. This way we could not only merge the two functions, but also have a convinient
# way to provide an autodiff-made acceleration when someone doesn't provide an `avv`.
# it would probably be very inefficient performace-wise for most cases, but it wouldn't hurt to have it somewhere
function levenberg_marquardt(df::OnceDifferentiable, initial_x::AbstractVector{T};
    x_tol::Real = 1e-8, g_tol::Real = 1e-12, maxIter::Integer = 1000,
    lambda = T(10), tau=T(Inf), lambda_increase::Real = 10.0, lambda_decrease::Real = 0.1,
    min_step_quality::Real = 1e-3, good_step_quality::Real = 0.75,
    show_trace::Bool = false, lower::Vector{T} = Array{T}(undef, 0), upper::Vector{T} = Array{T}(undef, 0), avv!::Union{Function,Nothing,Avv} = nothing
    ) where T

    # First evaluation
    value_jacobian!!(df, initial_x)
    
    if isfinite(tau)
        lambda = tau*maximum(jacobian(df)'*jacobian(df))
    end


    # check parameters
    ((isempty(lower) || length(lower)==length(initial_x)) && (isempty(upper) || length(upper)==length(initial_x))) ||
            throw(ArgumentError("Bounds must either be empty or of the same length as the number of parameters."))
    ((isempty(lower) || all(initial_x .>= lower)) && (isempty(upper) || all(initial_x .<= upper))) ||
            throw(ArgumentError("Initial guess must be within bounds."))
    (0 <= min_step_quality < 1) || throw(ArgumentError(" 0 <= min_step_quality < 1 must hold."))
    (0 < good_step_quality <= 1) || throw(ArgumentError(" 0 < good_step_quality <= 1 must hold."))
    (min_step_quality < good_step_quality) || throw(ArgumentError("min_step_quality < good_step_quality must hold."))


    # other constants
    MAX_LAMBDA = 1e16 # minimum trust region radius
    MIN_LAMBDA = 1e-16 # maximum trust region radius
    MIN_DIAGONAL = 1e-6 # lower bound on values of diagonal matrix used to regularize the trust region step


    converged = false
    x_converged = false
    g_converged = false
    iterCt = 0
    x = copy(initial_x)
    delta_x = copy(initial_x)
    a = similar(x)

    trial_f = similar(value(df))
    residual = sum(abs2, value(df))

    # Create buffers
    n = length(x)
    m = length(value(df))
    JJ = Matrix{T}(undef, n, n)
    n_buffer = Vector{T}(undef, n)
    Jdelta_buffer = similar(value(df))

    # and an alias for the jacobian
    J = jacobian(df)
    dir_deriv = Array{T}(undef,m)
    v = Array{T}(undef,n)

    # Maintain a trace of the system.
    tr = OptimizationTrace{LevenbergMarquardt}()
    if show_trace
        d = Dict("lambda" => lambda)
        os = OptimizationState{LevenbergMarquardt}(iterCt, sum(abs2, value(df)), NaN, d)
        push!(tr, os)
        println(os)
    end

    while (~converged && iterCt < maxIter)
        # jacobian! will check if x is new or not, so it is only actually
        # evaluated if x was updated last iteration.
        jacobian!(df, x) # has alias J

        # we want to solve:
        #    argmin 0.5*||J(x)*delta_x + f(x)||^2 + lambda*||diagm(J'*J)*delta_x||^2
        # Solving for the minimum gives:
        #    (J'*J + lambda*diagm(DtD)) * delta_x == -J' * f(x), where DtD = sum(abs2, J,1)
        # Where we have used the equivalence: diagm(J'*J) = diagm(sum(abs2, J,1))
        # It is additionally useful to bound the elements of DtD below to help
        # prevent "parameter evaporation".

        DtD = vec(sum(abs2, J, dims=1))
        for i in 1:length(DtD)
            if DtD[i] <= MIN_DIAGONAL
                DtD[i] = MIN_DIAGONAL
            end
        end

        # delta_x = ( J'*J + lambda * Diagonal(DtD) ) \ ( -J'*value(df) )
        mul!(JJ, transpose(J), J)
        @simd for i in 1:n
            @inbounds JJ[i, i] += lambda * DtD[i]
        end
        #n_buffer is delta C, JJ is g compared to Mark's code
        mul!(n_buffer, transpose(J), value(df))
        rmul!(n_buffer, -1)

        v .= JJ \ n_buffer


        if avv! != nothing
            #GEODESIC ACCELERATION PART
            avv!(dir_deriv, x, v)
            mul!(a, transpose(J), dir_deriv)
            rmul!(a, -1) #we multiply by -1 before the decomposition/division
            LAPACK.potrf!('U', JJ) #in place cholesky decomposition
            LAPACK.potrs!('U', JJ, a) #divides a by JJ, taking into account the fact that JJ is now the `U` cholesky decoposition of what it was before
            rmul!(a, 0.5)
            delta_x .= v .+ a
            #end of the GEODESIC ACCELERATION PART
        else
            delta_x = v
        end





        # apply box constraints
        if !isempty(lower)
            @simd for i in 1:n
               @inbounds delta_x[i] = max(x[i] + delta_x[i], lower[i]) - x[i]
            end
        end
        if !isempty(upper)
            @simd for i in 1:n
               @inbounds delta_x[i] = min(x[i] + delta_x[i], upper[i]) - x[i]
            end
        end

        # if the linear assumption is valid, our new residual should be:
        mul!(Jdelta_buffer, J, delta_x)
        Jdelta_buffer .= Jdelta_buffer .+ value(df)
        predicted_residual = sum(abs2, Jdelta_buffer)

        # try the step and compute its quality
        # compute it inplace according to NLSolversBase value(obj, cache, state)
        # interface. No bang (!) because it doesn't update df besides mutating
        # the number of f_calls

        # re-use n_buffer
        n_buffer .= x .+ delta_x
        value(df, trial_f, n_buffer)

        # update the sum of squares
        trial_residual = sum(abs2, trial_f)

        # step quality = residual change / predicted residual change
        rho = (trial_residual - residual) / (predicted_residual - residual)
        if trial_residual < residual && rho > min_step_quality
            # apply the step to x - n_buffer is ready to be used by the delta_x
            # calculations after this step.
            x .= n_buffer
            # There should be an update_x_value to do this safely
            copyto!(df.x_f, x)
            copyto!(value(df), trial_f)
            residual = trial_residual
            if rho > good_step_quality
                # increase trust region radius
                lambda = max(lambda_decrease*lambda, MIN_LAMBDA)
            end
        else
            # decrease trust region radius
            lambda = min(lambda_increase*lambda, MAX_LAMBDA)
        end

        iterCt += 1

        # show state
        if show_trace
            g_norm = norm(J' * value(df), Inf)
            d = Dict("g(x)" => g_norm, "dx" => delta_x, "lambda" => lambda)
            os = OptimizationState{LevenbergMarquardt}(iterCt, sum(abs2, value(df)), g_norm, d)
            push!(tr, os)
            println(os)
        end

        # check convergence criteria:
        # 1. Small gradient: norm(J^T * value(df), Inf) < g_tol
        # 2. Small step size: norm(delta_x) < x_tol
        if norm(J' * value(df), Inf) < g_tol
            g_converged = true
        end
        if norm(delta_x) < x_tol*(x_tol + norm(x))
            x_converged = true
        end
        converged = g_converged | x_converged
    end

    MultivariateOptimizationResults(
        LevenbergMarquardt(),    # method
        initial_x,             # initial_x
        x,                     # minimizer
        sum(abs2, value(df)),       # minimum
        iterCt,                # iterations
        !converged,            # iteration_converged
        x_converged,           # x_converged
        0.0,                   # x_tol
        0.0,
        false,                 # f_converged
        0.0,                   # f_tol
        0.0,
        g_converged,           # g_converged
        g_tol,                  # g_tol
        0.0,
        false,                 # f_increased
        tr,                    # trace
        first(df.f_calls),               # f_calls
        first(df.df_calls),               # g_calls
        0                      # h_calls
    )
end
