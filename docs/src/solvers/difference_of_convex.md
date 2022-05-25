# Difference of Convex Algorithm

This page is a first sketch how to model the following algorithm

Given the task to minimize

```math
    \operatorname*{argmin}_{p∈\mathcal M} f(p) - g(p)
```

This page is for now just collecting the involved functions, for example

## Difference of Convex Plan

```@autodocs
Modules = [Manopt]
Pages = ["plans/difference_of_convex_plan.jl"]
Order = [:type,:function]
Private = true
```

## Difference of Convex Solver

```@autodocs
Modules = [Manopt]
Pages = ["solvers/difference_of_convex.jl"]
Order = [:type,:function]
Private = true
```