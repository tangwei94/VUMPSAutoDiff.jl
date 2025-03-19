# TODO. wrap AC, C, AL, AR, T, EL, ER into a single struct. (which turns out to be a duplicate of what MPSKit.jl is doing....  )

"""
    VOMPSOptions

Configuration struct for VOMPS algorithm.

# Fields
- `tol::Float64`: Convergence tolerance for VOMPS updates
- `maxiter::Int`: Maximum number of iterations
- `verbosity::Int`: Output verbosity level (0 = silent, 1 = basic, 2 = detailed)
- `do_gauge_fixing::Bool`: Enable/disable gauge fixing after convergence
"""
struct VOMPSOptions
    tol::Float64
    maxiter::Int
    verbosity::Int
    do_gauge_fixing::Bool
end

"""
    VOMPSOptions(;tol=1e-12, maxiter=100, verbosity=1, do_gauge_fixing=true)

Construct VOMPSOptions with default values. Keyword arguments match field names.
"""
function VOMPSOptions(;tol=1e-12, maxiter=100, verbosity=1, do_gauge_fixing=true)
    VOMPSOptions(tol, maxiter, verbosity, do_gauge_fixing)
end

"""
    vomps_update!(AC1, C1, AL1, AR1, T, AL, AR)

Perform in-place VOMPS iteration step updating AC and C tensors using environment tensors.

# Arguments
- `AC1::MPSTensor`: Target AC tensor to update (modified in-place)
- `C1::MPSBondTensor`: Target C tensor to update (modified in-place)
- `AL1/AR1::MPSTensor`: Current left/right MPS tensors
- `T::MPOTensor`: MPO tensor for the Hamiltonian
- `AL/AR::MPSTensor`: Previous left/right MPS tensors
- `AC/C::MPSTensor`: Previous AC and C tensors
# Returns
- Tuple of updated (AC1, C1) tensors
"""
function vomps_update!(AC1::MPSTensor, C1::MPSBondTensor, AL1::MPSTensor, AR1::MPSTensor, EL::EnvTensorL, ER::EnvTensorR, T::MPOTensor, AL::MPSTensor, AR::MPSTensor, AC::MPSTensor, C::MPSBondTensor)

    TM_L = MPSMPOMPSTransferMatrix(AL1, T, AL)
    TM_R = MPSMPOMPSTransferMatrix(AR1, T, AR)

    _ = left_env!(EL, TM_L)
    λ = right_env!(ER, TM_R)

    # update AC and C
    α = norm(C) # if we don't add this, norm(AC) and norm(C) may decrease to zero during power iteration. adding this does not affect the final result, but makes the calculation more stable.
    @tensor AC1[-1 -2; -3] = (1/α) * (1/λ) * EL[-1; 1 2] * AC[1 3; 4] * T[2 -2; 3 5] * ER[4 5; -3]
    @tensor C1[-1; -2] = (1/α) * EL[-1; 1 3] * C[1; 2] * ER[2 3; -2]

    return AC1, C1
end
@non_differentiable vomps_update!(
    AC1::MPSTensor, C1::MPSBondTensor, 
    AL1::MPSTensor, AR1::MPSTensor,
    EL::EnvTensorL, ER::EnvTensorR, 
    T::MPOTensor, AL::MPSTensor, AR::MPSTensor, 
    AC::MPSTensor, C::MPSBondTensor
)

"""
    vomps!(AL1, AR1, T, AL, AR, opts)

MPO-MPS product compression using VOMPS algorithm.
The MPO is represented by the MPO tensor T, and the MPS is represented by the left and right MPS tensors AL and AR.
The output is saved in AL1 and AR1.

# Arguments
- `AL1/AR1::MPSTensor`: Target left/right MPS tensors to update (modified in-place)
- `T::MPOTensor`: MPO local tensor  
- `AL/AR::MPSTensor`: MPS tensors
- `opts::VOMPSOptions`: Algorithm configuration parameters

# Returns
- `AL1/AR1::MPSTensor`: Updated left/right MPS tensors (modified in-place)
- `AC1::MPSTensor`: Updated center AC tensor
- `C1::MPSBondTensor`: Updated center bond tensor
- `power_method_conv::Real`: Convergence metric from final gauge fixing (NaN if disabled). Measures the difference between the MPSs represented by AL1 and AL (should be = 0 for equivalent MPS)
"""
function vomps!(AL1::MPSTensor, AR1::MPSTensor, T::MPOTensor, AL::MPSTensor, AR::MPSTensor, AC::MPSTensor, C::MPSBondTensor, opts::VOMPSOptions)
    # initialize AC1 and C1
    AC1 = similar(AL1, space(AL1))
    C1 = similar(AL1, space(AL1, 1), space(AL1, 1))

    # VOMPS iterations
    EL = left_env(MPSMPOMPSTransferMatrix(AL1, T, AL))
    ER = right_env(MPSMPOMPSTransferMatrix(AR1, T, AR))
    conv_meas = Inf
    num_iter = 0
    for ix in 1:opts.maxiter
        vomps_update!(AC1, C1, AL1, AR1, EL, ER, T, AL, AR, AC, C)
        num_iter += 1
        conv_meas = mps_update!(AL1, AR1, AC1, C1)
        if opts.verbosity > 1
            printstyled("VOMPS iteration $ix: conv_meas = $conv_meas\r", color=:green)
        end
        if conv_meas < opts.tol
            break
        end
    end
    (opts.verbosity == 1) && printstyled("VOMPS final convergence measure: $conv_meas\n", color=:green)

    if opts.do_gauge_fixing
        power_method_conv = gauge_fix!(AL1, AR1, AC1, C1, AL)
    else
        # this does not work as well as the convergence measure from `gauge_fixing!`. 
        # This one is also more expensive.
        power_method_conv = abs(1 - mps_fidelity(AL1, AL))
    end

    return AL1, AR1, AC1, C1, power_method_conv, num_iter
end
@non_differentiable vomps!(AL1::MPSTensor, AR1::MPSTensor, T::MPOTensor, AL::MPSTensor, AR::MPSTensor, AC::MPSTensor, C::MPSBondTensor, opts::VOMPSOptions)

