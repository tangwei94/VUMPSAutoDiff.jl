# VUMPSAutoDiff

<!---
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://tangwei94.github.io/VUMPSAutoDiff.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tangwei94.github.io/VUMPSAutoDiff.jl/dev/)
--->
[![Build Status](https://github.com/tangwei94/VUMPSAutoDiff.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tangwei94/VUMPSAutoDiff.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/tangwei94/VUMPSAutoDiff.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/tangwei94/VUMPSAutoDiff.jl)

A minimal implementation of automatic differentiation for variational uniform matrix product states (VUMPS). 
This package provides a implementation of the algorithm in [arXiv:2304.01551](https://arxiv.org/abs/2304.01551) using [TensorKit.jl](https://github.com/jutho/TensorKit.jl).

This package mainly handles the following task:

Given a matrix product operator (MPO) represented by a `MPOTensor` `T`, the VUMPS algorithm looks for its fixed point matrix product state (MPS), which will be represented by a `MPSTensor` `A`.
The MPO usually comes from a partition function of a classical statistical model or the overlap of two projected entangled pair states (PEPS).
This package provides an implementation of the backward rule for the VUMPS algorithm, which is compatible with [Zygote.jl](https://github.com/FluxML/Zygote.jl), hence the name `VUMPSAutoDiff`.
More specifically, suppose in the forward computation, the final quantity of interest `y` will be a function of `A`, and suppose the derivative dy / dA is known, this package allows one to compute dy / dT using backward differentiation.

A minimal example of using this package is provided in `example.jl`, which computes the gradient of the free energy of Ising model with respect to the inverse temperature. 