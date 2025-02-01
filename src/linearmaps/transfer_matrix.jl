"""
    AbstractLinearMap

Abstract type representing linear maps used in transfer matrix calculations. 
Subtypes should implement these required methods:

1. **Vector space definitions**:
   - `right_space(TM)`: Returns the vector space for right environment calculations
   - `left_space(TM)`: Returns the vector space for left environment calculations

2. **Transfer operations**:
   - `left_transfer(TM, v)`: Apply transfer matrix to left environment vector `v`
   - `right_transfer(TM, v)`: Apply transfer matrix to right environment vector `v`

3. **Automatic differentiation**:
   - `ChainRulesCore.rrule` constructor: Defines gradient propagation rules

Concrete implementations can be seen in:
- `MPSMPSTransferMatrix` (MPS-MPS systems)
- `MPSMPOMPSTransferMatrix` (MPS-MPO-MPS systems) 
- `ACMap` (the linear map for AC in VUMPS iteration)

Subtypes must implement the core transfer operations to enable both forward 
calculations and backward gradient propagation through environment tensors.
"""
abstract type AbstractLinearMap end
#abstract type AbstractLinearMapBackward end

"""
    LinearMapBackward(VLs, VRs)

Storage structure for backward pass information of linear maps. Contains:
- `VLs`: Vector of left environment tensors
- `VRs`: Vector of right environment tensors

Used in automatic differentiation to accumulate gradients through transfer matrix operations.
"""
struct LinearMapBackward
    VLs::Vector{<:AbstractTensorMap}
    VRs::Vector{<:AbstractTensorMap}
end


Base.:+(bTM1::LinearMapBackward, bTM2::LinearMapBackward) = LinearMapBackward([bTM1.VLs; bTM2.VLs], [bTM1.VRs; bTM2.VRs])
Base.:-(bTM1::LinearMapBackward, bTM2::LinearMapBackward) = LinearMapBackward([bTM1.VLs; bTM2.VLs], [bTM1.VRs; -1*bTM2.VRs])
Base.:*(a::Number, bTM::LinearMapBackward) = LinearMapBackward(bTM.VLs, a * bTM.VRs)
Base.:*(bTM::LinearMapBackward, a::Number) = LinearMapBackward(bTM.VLs, a * bTM.VRs)

"""
    right_env(TM::AbstractLinearMap) -> AbstractTensorMap

Calculate the right environment tensor (ρr) of a transfer matrix by finding the 
dominant right eigenvector of the transfer operator.

# Arguments
- `TM`: Abstract linear map representing the transfer matrix

# Returns
- Right environment tensor (fixed point of right transfer operation)
"""
function right_env(TM::AbstractLinearMap)
    init = rand(ComplexF64, right_space(TM))
    _, ρrs, _ = eigsolve(v -> right_transfer(TM, v), init, 1, :LM)
    return ρrs[1]
end

"""
    left_env(TM::AbstractLinearMap) -> AbstractTensorMap

Calculate the left environment tensor (ρl) of a transfer matrix by finding the 
dominant left eigenvector of the transfer operator.

# Arguments
- `TM`: Abstract linear map representing the transfer matrix

# Returns
- Left environment tensor (fixed point of left transfer operation)
"""
function left_env(TM::AbstractLinearMap)
    init = rand(ComplexF64, left_space(TM))
    _, ρls, _ = eigsolve(v -> left_transfer(TM, v), init, 1, :LM)
    return ρls[1]
end

"""
    right_env_backward(TM, λ, vr, ∂vr) -> AbstractTensorMap

Backward pass computation for right environment calculation. Solves the linear system:
(TM' - λ*I)ξr = ∂vr to compute gradients.

# Arguments
- `TM`: Original transfer matrix
- `λ`: Dominant eigenvalue from forward pass
- `vr`: Right environment vector from forward pass
- `∂vr`: Gradient from subsequent computations

# Returns
- Gradient contribution ξr for the transfer matrix
"""
function right_env_backward(TM::AbstractLinearMap, λ::Number, vr::AbstractTensorMap, ∂vr::AbstractTensorMap)
    init = similar(vr)
    randomize!(init); 
    init = init - dot(vr, init) * vr # the subtracted part lives in the null space of flip(TM) - λ*I
    
    err = norm(dot(vr, ∂vr))
    tol = 1e-9 * max(1, norm(∂vr))
    (err > tol) && @warn "right_env_backward: forward computation not gauge invariant: final computation should not depend on the phase of vr. err=$err, tol=$tol" 
    ∂vr = ∂vr - dot(vr, ∂vr) * vr 
    ξr, info = linsolve(x -> left_transfer(TM, x) - λ*x, ∂vr', init') # ξr should live in the space of vl
    (info.converged == 0) && @warn "right_env_backward not converged: normres = $(info.normres)"
    
    return ξr
end

"""
    left_env_backward(TM, λ, vl, ∂vl) -> AbstractTensorMap

Backward pass computation for left environment calculation. Solves the linear system:
(TM - λ*I)ξl = ∂vl to compute gradients.

# Arguments
- `TM`: Original transfer matrix
- `λ`: Dominant eigenvalue from forward pass
- `vl`: Left environment vector from forward pass
- `∂vl`: Gradient from subsequent computations

# Returns
- Gradient contribution ξl for the transfer matrix
"""
function left_env_backward(TM::AbstractLinearMap, λ::Number, vl::AbstractTensorMap, ∂vl::AbstractTensorMap)
    init = similar(vl); 
    randomize!(init); 
    init = init - dot(vl, init) * vl # the subtracted part lives in the null space of TM - λ*I

    err = norm(dot(vl, ∂vl))
    tol = 1e-9 * max(1, norm(∂vl))
    (err > tol) && @warn "left_env_backward: forward computation not gauge invariant: final computation should not depend on the phase of vl. err=$err, tol=$tol" 
    ∂vl = ∂vl - dot(vl, ∂vl) * vl 
    ξl, info = linsolve(x -> right_transfer(TM, x) - λ*x, ∂vl', init') # ξl should live in the space of vr
    (info.converged == 0) && @warn "left_env_backward not converged: normres = $(info.normres)"

    return ξl
end

"""
    ChainRulesCore.rrule(::typeof(right_env), TM)

Custom reverse rule for right environment calculation using ChainRules.jl. 
Computes gradients through the eigen solver using the backward pass solution.

# Returns
- Right environment tensor and pullback function
"""
function ChainRulesCore.rrule(::typeof(right_env), TM::AbstractLinearMap)
    init = rand(ComplexF64, right_space(TM))
    λrs, vrs, _ = eigsolve(v -> right_transfer(TM, v), init, 1, :LM)
    λr, vr = λrs[1], vrs[1]

    function right_env_pushback(_∂vr)
        ∂vr = unthunk(_∂vr)
        ξr = right_env_backward(TM, λr, vr, ∂vr)
        return NoTangent(), LinearMapBackward([-ξr], [vr])
    end
    return vr, right_env_pushback
end

"""
    ChainRulesCore.rrule(::typeof(left_env), TM)

Custom reverse rule for left environment calculation using ChainRules.jl. 
Computes gradients through the eigen solver using the backward pass solution.

# Returns
- Left environment tensor and pullback function
"""
function ChainRulesCore.rrule(::typeof(left_env), TM::AbstractLinearMap)
    init = rand(ComplexF64, left_space(TM))
    λls, vls, _ = eigsolve(v -> left_transfer(TM, v), init, 1, :LM)
    λl, vl = λls[1], vls[1]
   
    function left_env_pushback(_∂vl)
        ∂vl = unthunk(_∂vl)
        ξl = left_env_backward(TM, λl, vl, ∂vl)
        return NoTangent(), LinearMapBackward([vl], [-ξl])
    end
    return vl, left_env_pushback
end
