@doc raw"""
    DifferenceOfConvexProblem <: Problem

Specify a problem for a [`difference_of_convex`](@ref) algorithm.

The problem is of the form

```math
    \operatorname*{argmin}_{p\in \mathcal M} f(p) - g(p)
```

where both ``f`` and ``g`` are convex, lsc. and proper.

# Fields

* `M`  – an `AbstractManifold`
* `cost` – an implementation of ``F(p) = f(p)-g(p)``
* `∂g!!` – a deterministic version of ``∂g: \mathcal M → T\mathcal M``,
  i.e. calling `∂g(p)` returns a subgradient of ``g`` at `p` and if there is more than one,
  it returns a deterministic choice.

Note that the subdifferential might be given in two possible signatures
* `∂g(M,p)` which does an [`AllocatingEvaluation`](@ref)
* `∂g!(M, X, p)` which does an [`MutatingEvaluation`](@ref) in place of `X`.
"""
struct DifferenceOfConvexProblem{T,TManifold<:AbstractManifold,TCost,TSubGrad}
    M::TManifold
    cost::TCost
    ∂g!!::TSubGrad
    function DifferenceOfConvexProblem(
        M::TManifold,
        cost::TC,
        ∂g::TSG;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
    ) where {TManifold<:AbstractManifold,TC,TSG}
        return new{typeof(evaluation),TManifold,TC,TSG}(M, cost, ∂g)
    end
end

"""
    get_subgradient(p, q)
    get_subgradient!(p, X, q)

Evaluate the (sub)gradient of a [`DifferenceOfConvexProblem`](@ref)` p` at the point `q`.

The evaluation is done in place of `X` for the `!`-variant.
The `T=`[`AllocatingEvaluation`](@ref) problem might still allocate memory within.
When the non-mutating variant is called with a `T=`[`MutatingEvaluation`](@ref)
memory for the result is allocated.
"""
function get_subgradient(P::DifferenceOfConvexProblem{AllocatingEvaluation}, q)
    return P.∂g!!(P.M, q)
end
function get_subgradient(P::DifferenceOfConvexProblem{MutatingEvaluation}, q)
    X = zero_vector(P.M, q)
    return P.∂g!!(P.M, X, q)
end
function get_subgradient!(P::DifferenceOfConvexProblem{AllocatingEvaluation}, X, q)
    return copyto!(P.M, X, P.∂g!!(P.M, q))
end
function get_subgradient!(P::DifferenceOfConvexProblem{MutatingEvaluation}, X, q)
    return P.∂g!!(P.M, X, q)
end

@doc raw"""
    DifferenceOfConvexOptions{Type} <: Options

A struct to store the current state of the algorithm as well as the form.
It comes in two forms, depending on the realisation of the `subproblem`.

# Fields

* `p` – the current iterate, i.e. a point on the manifold
* `X` – the current subgradient, i.e. a tangent vector to `p`.
* `subtask` – a type representing the subtask.

For the subtask, we need a method to solve

```math
    \operatorname*{argmin}_{q∈\mathcal M} f(p) - ⟨X, \log_p q⟩
```

where currently two variants are supported

1. `subtask(M, q, X, p)` is a mutating function, i.e. we have a closed form solution of the
  optimization problem given `M`, `X` and `p` which is computed in place of `q`, which even
  works correctly, if we pass the same memory to `p` and `q`.
2. `subtask::Tuple{<:Problem,<:Options}` specifies a plan to solve the sub task with a subsolver,
  i.e. the cost within `subtask[1]` is a [`DifferenceInner`](@ref) using `p`and `X`
  internally, i.e. the cost is updated as soon as they are updated.
  Similarly for gradient based functions using the [`DifferenceInnerGrad`](@ref).
"""
mutable struct DifferenceOfConvexOptions{S,P,T,SC<:StoppingCriterion}
    p::P
    X::T
    subtask::S
    stop::SC
    function DifferenceOfConvexOptions(
        M::AbstractManifold,
        p::P,
        subtask::S;
        initial_vector::T = zero_vector(M, p),
        stopping_criterion::SC = StopAfterIteration(200),
    ) where {P,S,T,SC<:StoppingCriterion}
        return new{S,T,P,SC}(p, initial_vector, subtask, stopping_criterion)
    end
end

@doc raw"""
    DifferenceInner

A functor `(M,p) → ℝ` to represent the inner problem of a [`DifferenceOfConvexProblem`](@ref),
i.e. a cost function of the form

```math
    F_{p,X}(q) = f(q) - ⟨X, \log_pq⟩
```
for a point `p` and a tangent vector `X` at `p` that are stored within this functor as well

# Fields

* `f` a function
* `p` a point on a manifold
* `X` a tangent vector at `p`
"""
mutable struct DifferenceInner{P,T,TG}
    f::TG
    p::P
    X::T
end
(F::DifferenceInner)(M, q) = F.f(q) - inner(M, F.p, F.X, log(M, F.p, q))

@doc raw"""
    DifferenceInnerGrad

A functor `(M,X,p) → ℝ` to represent the gradient of the inner problem of a [`DifferenceOfConvexProblem`](@ref),
i.e. for a cost function of the form

```math
    F_{p,X}(q) = f(q) - ⟨X, \log_pq⟩
```

its gradient is given by using ``F=F_1(F_2(q))``, where ``F_1(Y) = ⟨X,Y⟩`` and ``F_2(q) = \log_pq``
and the chain rule as well as the [`adjoint_differential_log_argument`](@ref) for ``D^*F_2(p)``

```math
    \operatorname{grad} F (q) = \operatorname{grad} f(q) - DF_2^*(q)[X]
```

for a point `p` and a tangent vector `X` at `p` that are stored within this functor as well

# Fields

* `grad_f` the gradient of ``f`` (see [`DifferenceInner`](@ref)) as
* `p` a point on a manifold
* `X` a tangent vector at `p`

# Constructor
    DifferenceInnerGrad(grad_f, p, X; evaluation=AllocatingEvaluation())

Where you specify whether `grad_f` is [`AllocatingEvaluation`](@ref) or [`MutatingEvaluation`](@ref),
while this function still provides _both_ signatures.
"""
mutable struct DifferenceInnerGrad{E<:AbstractEvaluationType,P,T,TG}
    grad_f::TG
    p::P
    X::T
    function DifferenceInnerGrad(
        grad_f::TG,
        p::P,
        X::T;
        evaluation::E = AllocatingEvaluation(),
    ) where {TG,P,T,E<:AbstractEvaluationType}
        return new{E,P,T,TG}(grad_f, p, X)
    end
end
function (grad_F::DifferenceInnerGrad{AllocatingEvaluation})(M, q)
    return grad_F.grad_f(M, q) .-
           adjoint_differential_log_argument(M, grad_F.p, q, grad_F.X)
end
function (grad_F::DifferenceInnerGrad{AllocatingEvaluation})(M, Y, q)
    copyto!(
        M,
        Y,
        q,
        grad_F.grad_f(M, q) .- adjoint_differential_log_argument(M, grad_F.p, q, grad_F.X),
    )
    return Y
end
function (grad_F!::DifferenceInnerGrad{MutatingEvaluation})(M, Y, q)
    grad_F!.grad_f(M, Y, p)
    Y .-= adjoint_differential_log_argument(M, grad_F!.p, q, grad_F!.X)
    return Y
end
function (grad_F!::DifferenceInnerGrad{MutatingEvaluation})(M, q)
    Y = zero_vector(M, q)
    grad_F!.grad_g(M, Y, p)
    Y .-= adjoint_differential_log_argument(M, grad_F!.p, q, grad_F!.X)
    return Y
end
