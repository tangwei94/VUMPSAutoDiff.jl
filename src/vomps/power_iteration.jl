function power_iteration_step!(subspace_ALs::Vector{MPSTensor}, subspace_errs::Vector{MPSTensor}, T::MPOTensor, AL::MPSTensor, AR::MPSTensor, AC::MPSTensor, C::MPSBondTensor; maintain_subspace_size::Bool=true, update_subspace::Bool=true)
    AL1, AR1 = deepcopy(AL), deepcopy(AR)
    AL1, AR1, AC1, C1, power_method_conv, num_iter = vomps!(AL1, AR1, T, AL, AR, AC, C, VOMPSOptions(maxiter=100, tol=1e-12, verbosity=1, do_gauge_fixing=true));

    if update_subspace
        push!(subspace_ALs, AL1)
        push!(subspace_errs, AL1 - AL)
        if maintain_subspace_size
            popfirst!(subspace_ALs)
            popfirst!(subspace_errs)
        end
    end

    return AL1, AR1, AC1, C1, power_method_conv, num_iter
end

struct MPOPowerIterationOptions
    M::Int
    ΔM::Int
    tol::Float64
    diis_criterion::Float64
    max_diis_step::Int
    damping_factor::Float64 
end

function MPOPowerIterationOptions(; M::Int = 10, ΔM::Int = 5, tol::Float64 = 1e-8, diis_criterion::Float64 = 1e-3, max_diis_step::Int = 10, damping_factor::Float64=1e-8)
    return MPOPowerIterationOptions(M, ΔM, tol, diis_criterion, max_diis_step, damping_factor)
end

function mpo_power_iterations(T::MPOTensor, AL::MPSTensor, AR::MPSTensor, AC::MPSTensor, C::MPSBondTensor, options::MPOPowerIterationOptions)

    subspace_errs = MPSTensor[]
    subspace_ALs = MPSTensor[]
    total_num_iter = 0

    power_method_conv = Inf
    while power_method_conv > options.diis_criterion
        AL, AR, AC, C, power_method_conv, num_iter = power_iteration_step!(subspace_ALs, subspace_errs, T, AL, AR, AC, C; maintain_subspace_size=false)
        total_num_iter += num_iter
        println("Power method convergence: $power_method_conv, total_num_iter: $total_num_iter")
    end

    for ix in 1:options.M
        AL, AR, AC, C, power_method_conv, num_iter = power_iteration_step!(subspace_ALs, subspace_errs, T, AL, AR, AC, C; maintain_subspace_size=false)
        total_num_iter += num_iter
        println("Power method convergence: $power_method_conv, total_num_iter: $total_num_iter")
        (power_method_conv < options.tol) && break
    end

    if options.max_diis_step > 0
        B = initialize_ovlpmat(subspace_errs; damping_factor=options.damping_factor, inner=inner)
        for diis_step in 1:options.max_diis_step
            Anew = DIIS_extrapolation(B, subspace_ALs)
            println("DIIS extrapolation step $diis_step")
            
            AL, R = left_canonical_QR(Anew; verbosity=1);
            AR, L = right_canonical_QR(Anew; verbosity=1);
            C = R * L;
            AC = AL * C;

            for _ in 1:options.ΔM
                AL, AR, AC, C, power_method_conv, num_iter = power_iteration_step!(subspace_ALs, subspace_errs, T, AL, AR, AC, C; maintain_subspace_size=true)
                total_num_iter += num_iter
                println("Power method convergence: $power_method_conv, total_num_iter: $total_num_iter")
                (power_method_conv < options.tol) && return (AL, AR, AC, C)
            end
            update_ovlpmat!(B, options.ΔM, subspace_errs; damping_factor=options.damping_factor, inner=inner)
        end
    end
    return (AL, AR, AC, C)
end

#T = tensor_square_ising(asinh(1) / 2);
#A = rand(ComplexF64, ℂ^4*ℂ^2, ℂ^4);
#AL, AR, AC, C = full_canonicalization(A);
#AL, AR, AC, C = mpo_power_iterations(T, AL, AR, AC, C, MPOPowerIterationOptions(;tol=1e-9)) ;
#vumps(T; A=AL, verbosity=1, tol=1e-9);