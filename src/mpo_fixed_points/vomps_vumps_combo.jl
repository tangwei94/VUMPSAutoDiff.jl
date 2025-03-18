struct VOMPSVUMPSComboOptions
    M::Int
    VUMPS_criterion::Float64
    tol::Float64
    maxiter::Int
end

function VOMPSVUMPSComboOptions(; M::Int = 10, VUMPS_criterion::Float64 = 1e-6, tol::Float64 = 1e-9, maxiter::Int = 100)
    return VOMPSVUMPSComboOptions(M, VUMPS_criterion, tol, maxiter)
end

function vumps_vomps_combo_iterations(T::MPOTensor, AL::MPSTensor, AR::MPSTensor, AC::MPSTensor, C::MPSBondTensor, options::VOMPSVUMPSComboOptions)

    total_num_iter = 0
    power_method_conv = Inf
    while power_method_conv > options.VUMPS_criterion
        AL1, AR1 = deepcopy(AL), deepcopy(AR)
        AL1, AR1, AC1, C1, power_method_conv, num_iter = vomps!(AL1, AR1, T, AL, AR, AC, C, VOMPSOptions(maxiter=250, tol=1e-12, verbosity=1, do_gauge_fixing=false));
        AL, AR, AC, C = AL1, AR1, AC1, C1

        total_num_iter += num_iter
        println("Power method convergence: $power_method_conv, total_num_iter: $total_num_iter")
        @show norm(AL), norm(AR), norm(AC), norm(C)
    end

    for ix in 1:options.maxiter
        for jx in 1:options.M
            AL1, AR1 = deepcopy(AL), deepcopy(AR)
            AL1, AR1, AC1, C1, power_method_conv, num_iter = vomps!(AL1, AR1, T, AL, AR, AC, C, VOMPSOptions(maxiter=250, tol=1e-12, verbosity=1, do_gauge_fixing=false));
            AL, AR, AC, C = AL1, AR1, AC1, C1
            
            total_num_iter += num_iter
            println("$ix, $jx, Power method convergence: $power_method_conv, total_num_iter: $total_num_iter")
            @show norm(AL), norm(AR), norm(AC), norm(C)
        end
        vumps_update!(AL, AR, AC, C, T)
        vumps_conv = mps_update!(AL, AR, AC, C)
        (options.M > 0) && (AL, AR, AC, C = full_canonicalization(AL))
        println("-- $ix, VUMPS convergence: $vumps_conv")
        if vumps_conv < options.tol
            break
        end
    end

    return AL, AR, AC, C
end