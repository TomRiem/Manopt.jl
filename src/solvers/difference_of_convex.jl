@doc raw"""
    difference_of_convex(M, cost, f, gradf, ∂g, p;
        initial_vector=zero_vector(M,p)
        evaluation::AbstractEvaluationType = MutatingEvaluation(),
        subtask = (
            GradientProblem(M,DifferenceInner(f, p, initial_vector), DifferenceInnerGrad()),
            GradientDescentOptions(M,copy(M,p)))
        stopping_criterion = StopAfterIteration(200),
    )

Compute the difference of convex algorithm to minimize


```math
    \operatorname*{argmin}_{p∈\mathcal M} f(p) - g(p)
```

This algorithm performs the following steps given a start point `p`= ``p^{(0)}``.
Then repeat for ``k=0,1,\ldots``

1. Take ``X^{(k)}  ∈ ∂g(p^{(k)}``
2. Set the next iterate to the solution of the subproblem
  ```math
    p^{(k+1)} \in \operatorname*{argmin}_{q\in \mathcal M} f(p) - f(q) - ⟨X^{(k)}, \log_{p^{(k)}}q⟩
  ```

until the `stopping_criterion` is fulfilled.

!!! note
    If the subproblem requires `p` and `X` it should be stored by reference so the update is implicit,
    cf. the default subtask in the signature above.
"""
function difference_of_convex(M::AbstractManifold, cost, f, grad_f, ∂g, p; kwargs...)
    q = copy(M, p)
    difference_of_convex!(M, cost, f, grad_f, ∂g, q; kwargs...)
    return q
end

@doc raw"""
    difference_of_convex!(M, cost, f, gradf, ∂g, p; kwargs...)

Run the difference of convex algorithm and perform the steps in place of `p`.
See ∞`difference_of_convex`](@ref) for more details.
"""
function difference_of_convex!(
    M::AbstractManifold,
    q,
    cost,
    f,
    grad_f,
    ∂g,
    p;
    initial_vector = zero_vector(M, p),
    evaluation::AbstractEvaluationType = MutatingEvaluation(),
    subtask = (
        GradientProblem(
            M,
            DifferenceInner(f, p, initial_vector),
            DifferenceInnerGrad(grad_f, p, initial_vector; evaluation = evaluation),
        ),
        GradientDescentOptions(M, copy(M, p)),
    ),
    stopping_criterion = StopAfterIteration(200),
    return_options = false,
    kwargs..., #collect rest
)
    P = DifferenceOfConvexProblem(M, cost, ∂g; evaluation = evaluation)
    O = DifferenceOfConvexOptions(
        M,
        p,
        subtask;
        stopping_criterion = stopping_criterion,
        initial_vector = initial_vector,
    )
    o = decorate_options(o; debug = debug, kwargs...)
    resultO = solve(p, o)
    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end
function initialize_solver!(::DifferenceOfConvexProblem, O::DifferenceOfConvexOptions)
    return O
end
function step_solver!(
    P::DifferenceOfConvexProblem,
    O::DifferenceOfConvexOptions{<:Tuple{<:Problem,<:Options}},
    iter,
)
    get_subgradient!(P, O.X, O.p) # evaluate ∂g(p), store the result in O.X
    solve(O.subtask[1], O.subtask[2]) # call the subsolver
    # copy result from subsolver to current iterate
    copyto!(M, O.p, get_solver_result(O.subtask[2]))
    return O
end
#
# Variant II: subtask is a mutating function providing a closed form soltuion
#
function step_solver!(P::DifferenceOfConvexProblem, O::DifferenceOfConvexOptions, iter)
    get_subgradient!(P, O.X, O.p) # evaluate ∂g(p), store the result in O.X
    O.subtask(M, O.p, O.X, O.p) # evaluate the closed form solution and store the result in O.p
    return O
end

get_solver_result(O::DifferenceOfConvexOptions) = O.p