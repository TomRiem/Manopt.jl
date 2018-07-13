"""
    Manopt.jl
A package to perform Optimization methods on manifold in Julia uncluding
high dimensional power manifolds to tacke manifold-valued image processing.

See `Readme.md` for an exaustive list of features and examples/ for several
examples that can just be `include`d.
"""
module Manopt
  using SimpleTraits
# Manifolds
  include("manifolds/Manifold.jl") #base type
  # Traits (properties/decorators)
  include("manifolds/traits/MatrixManifold.jl")
  include("manifolds/traits/LieGroup.jl")
  # specific manifolds
  include("manifolds/Circle.jl")
	include("manifolds/Euclidean.jl")
	include("manifolds/SymmetricPositiveDefinite.jl")
  include("manifolds/PowerManifold.jl")
  include("manifolds/ProductManifold.jl")
  include("manifolds/Sphere.jl")
  # Functions
  include("functions/AdjointJacobiFields.jl")
  include("functions/gradients.jl")
  include("functions/JacobiFields.jl")
  include("functions/proximalMaps.jl")
  # ...corresponding plans consisting of problems and options
  include("plans/problem.jl")
  include("plans/options.jl")
  # ...solvers
  include("solvers/cyclicProximalPoint.jl")
  include("solvers/steepestDescent.jl")
  # algorithms
  include("algorithms/basicAlgorithms.jl")
  include("algorithms/lineSearch.jl")
  # Plots
  include("plots/SpherePlots.jl")
  # helpers
  include("helpers/imageHelpers.jl")
  include("helpers/debugFunctions.jl")
  # data
  include("data/artificialDataFunctions.jl")
end