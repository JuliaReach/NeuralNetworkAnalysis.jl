module NeuralNetworkAnalysis

include("init.jl")
include("problem.jl")
include("nnops.jl")
include("setops.jl")
include("split.jl")
include("utils.jl")
include("simulate.jl")
include("solve.jl")

# problem types
export ControlledPlant,
       BlackBoxController

# splitters
export BoxSplitter, ZonotopeSplitter,
       IndexedSplitter,
       SignSplitter

# solvers
export solve, forward, simulate,
       SampledApprox, VertexSolver, BoxSolver, SplitSolver, BlackBoxSolver

# utility functions
export @modelpath, read_nnet_mat, read_nnet_yaml, print_timed

end
