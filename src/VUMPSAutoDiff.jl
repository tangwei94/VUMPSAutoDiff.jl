module VUMPSAutoDiff

__precompile__(true)

# Write your package code here.
using LinearAlgebra, VectorInterface
using TensorKit, TensorOperations, KrylovKit, TensorKitManifolds
using ChainRules, ChainRulesCore, Zygote
using OptimKit
using JLD2

export randomize!
export AbstractLinearMap, LinearMapBackward, left_transfer, right_transfer 
export MPSMPSTransferMatrix, MPSMPOMPSTransferMatrix
export ACMap, fixed_point
export right_env, left_env
export gauge_fixing, overall_u1_phase
export mps_update!, mps_update, vumps_update, vumps

export vomps!, VOMPSOptions

export DIIS_extrapolation_alg, power_method_alg, iterative_solver

include("utils.jl");

# Linear maps
include("linearmaps/transfer_matrix.jl");
include("linearmaps/MPSMPSTransferMatrix.jl");
include("linearmaps/MPSMPOMPSTransferMatrix.jl");
include("linearmaps/ACMap.jl");

include("canonicalization.jl");
include("vumps.jl");

# DIIS tools for speeding up iterative solvers
include("toolbox/diis.jl")
include("toolbox/gauge_fixing.jl"); 

include("vomps.jl");

end
