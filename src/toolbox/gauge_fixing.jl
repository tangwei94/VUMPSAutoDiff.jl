"""
    gauge_fixing(AL1, AL2)

Compute the optimal gauge transformation between two left-canonical MPS tensors by diagonalizing
their transfer matrix. Returns the unitary transformation and convergence metric.

# Arguments
- `AL1`: Reference left-canonical MPS tensor
- `AL2`: Target MPS tensor to be gauge-fixed (will be transformed to match AL1's gauge)

# Returns
- `U::AbstractTensorMap`: Unitary transformation matrix satisfying ``AL1 ≈ U * AL2 * U' ``
- `conv_meas::Real`: if AL1 and AL2 are equivalent up to a unitary transformation, this should be 0. This is a measure of the deviation between the MPS's represented by AL1 and AL2
"""
function gauge_fixing(AL1::AbstractTensorMap, AL2::AbstractTensorMap)
    TM = MPSMPSTransferMatrix(AL1, AL2)
    σ = left_env(TM)
    U, R = leftorth(σ; alg=QRpos())

    rmul!(R, dim(space(R, 1))/tr(R))  # Normalize R matrix
    conv_meas = norm(R - id(space(R, 1)))  # Measures deviation from identity

    return U, conv_meas
end
@non_differentiable gauge_fixing(args...)

"""
    overall_u1_phase(T1, T2)

Compute the global U(1) phase difference between two tensors. Returns a complex phase factor
λ such that T1 ≈ λ*T2 up to numerical precision. Uses the trace of T1'*T2 to determine the phase.

# Arguments
- `T1`, `T2`: Input tensors to compare

# Returns
- Complex phase factor λ with |λ| = 1
"""
function overall_u1_phase(T1::AbstractTensorMap, T2::AbstractTensorMap)
    T1T2trace = tr(T2' * T1)
    return (norm(T1)/norm(T2)) * T1T2trace / abs(T1T2trace)
end
#function overall_u1_phase(T1::AbstractTensorMap, T2::AbstractTensorMap)
#    for (f1, f2) in fusiontrees(T2)
#        for ix in 1:length(T2[f1, f2])
#            if norm(T2[f1, f2][ix]) > 1e-2
#                return T1[f1, f2][ix] / T2[f1, f2][ix]
#                @show norm(T1[f1, f2][ix] / T2[f1, f2][ix]) 
#            end
#        end
#    end
#end
#function ChainRulesCore.rrule(::typeof(overall_u1_phase), T1::AbstractTensorMap, T2::AbstractTensorMap)
#    info = []
#    for (f1, f2) in fusiontrees(T2)
#        for ix in 1:length(T2[f1, f2])
#            if norm(T2[f1, f2][ix]) > 1e-2
#                fs = (f1, f2)
#                index = ix
#                α = T1[f1, f2][ix] / T2[f1, f2][ix]
#                push!(info, (fs, index, α))
#                break
#            end
#        end
#    end
#    fs, index, α = info[1] 
#
#    function overall_u1_phase_pushback(∂α)
#        ∂T1 = zero(T1)
#        ∂T2 = zero(T2)
#        ∂T1[fs...][index] = ∂α / T2[fs...][index]
#        ∂T2[fs...][index] = -∂α * T1[fs...][index] / T2[fs...][index]^2
#        return NoTangent(), ∂T1, ∂T2
#    end
#
#    return α, overall_u1_phase_pushback
#end
@non_differentiable overall_u1_phase(::AbstractTensorMap, ::AbstractTensorMap)

"""
    gauge_fix!(AL2, AL1)

In-place gauge fixing of left-canonical MPS tensor AL2 to match AL1's gauge. Performs:
1. Unitary gauge transformation using `gauge_fixing`
2. Global U(1) phase correction using `overall_u1_phase`

# Arguments
- `AL2`: left-canonical MPS tensor to be modified in-place (will be gauge fixed to match AL1's gauge)
- `AL1`: Reference left-canonical MPS tensor in the desired gauge

# Returns
- `conv_meas::Real`: Convergence metric from gauge fixing, measures residual difference between 
  the gauge-transformed AL2 and AL1 (should be ≈ 0 for equivalent MPS)
"""
function gauge_fix!(AL2::MPSTensor, AR2::MPSTensor, AC2::MPSTensor, C2::MPSBondTensor, AL1::MPSTensor)
    U, conv_meas = gauge_fixing(AL1, AL2)
    @tensor AL2[-1 -2; -3] = AL2[1 -2; 2] * U[-1; 1] * U'[2; -3] 
    @tensor AR2[-1 -2; -3] = AR2[1 -2; 2] * U[-1; 1] * U'[2; -3] 
    @tensor AC2[-1 -2; -3] = AC2[1 -2; 2] * U[-1; 1] * U'[2; -3] 
    @tensor C2[-1; -2] = C2[1 ;2] * U[-1; 1] * U'[2; -2] 
    λ = overall_u1_phase(AL1, AL2)
    rmul!(AL2, λ)
    rmul!(AR2, λ)
    rmul!(AC2, λ)
    return conv_meas 
end
@non_differentiable gauge_fix!(AL2::MPSTensor, AR2::MPSTensor, AC2::MPSTensor, C2::MPSBondTensor, AL1::MPSTensor)

"""
    gauge_fixed_vumps_iteration(AL, AR, T)

Perform a single gauge-fixed VUMPS iteration. Returns updated (AL, AR) tensors after:
1. VUMPS update to get (AC, C)
2. MPS update to get new (AL, AR)
3. Gauge fixing with unitary transformation
4. Global phase correction

# Arguments
- `AL`, `AR`: Current left/right MPS tensors
- `T`: MPO tensor for the Hamiltonian

# Returns
- Tuple of (AL1_gauged, AR1_gauged) updated tensors
"""
function gauge_fixed_vumps_iteration(AL::MPSTensor, AR::MPSTensor, T::MPOTensor)
    AC1, C1 = vumps_update(AL, AR, T)
    AL1, AR1, _ = mps_update(AC1, C1)

    U, _ = gauge_fixing(AL, AL1)
    @tensor AR1_gauged[-1 -2; -3] := AR1[1 -2; 2] * U[-1; 1] * U'[2; -3]
    @tensor AL1_gauged[-1 -2; -3] := AL1[1 -2; 2] * U[-1; 1] * U'[2; -3]

    λ = overall_u1_phase(AL, AL1_gauged)
    AL1_gauged = AL1_gauged * λ
    AR1_gauged = AR1_gauged * λ

    return AL1_gauged, AR1_gauged 
end

"""
    ordinary_vumps_iteration(AL, AR, T)

Perform a standard VUMPS iteration without gauge fixing. Returns updated (AL, AR) tensors
through basic VUMPS update steps.

# Arguments
- `AL`, `AR`: Current left/right MPS tensors  
- `T`: MPO tensor for the Hamiltonian

# Returns 
- Tuple of (AL1, AR1) updated tensors
"""
function ordinary_vumps_iteration(AL::MPSTensor, AR::MPSTensor, T::MPOTensor)
    AC1, C1 = vumps_update(AL, AR, T)
    AL1, AR1, _ = mps_update(AC1, C1)

    return AL1, AR1 
end