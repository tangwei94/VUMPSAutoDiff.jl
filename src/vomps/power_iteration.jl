
T = tensor_square_ising(asinh(1) / 2);
A = rand(ComplexF64, ℂ^2*ℂ^2, ℂ^2);
AL, R = VUMPSAutoDiff.left_canonical_QR(A; verbosity=2);
AR, L = VUMPSAutoDiff.right_canonical_QR(A; verbosity=2);

C = R * L;

AC = AL * C;
@tensor AC1[-1 -2; -3] := AR[1 -2; -3] * C[-1; 1];
@show norm(AC - AC1)

subspace_errs = VUMPSAutoDiff.MPSTensor[]
subspace_ALs = VUMPSAutoDiff.MPSTensor[]

function power_iteration_step!(subspace_ALs, subspace_errs, T, AL, AR, AC, C; maintain_subspace_size::Bool=true)
    AL1, AR1 = deepcopy(AL), deepcopy(AR)
    AL1, AR1, AC1, C1, power_method_conv = vomps!(AL1, AR1, T, AL, AR, AC, C, VOMPSOptions(maxiter=100, tol=1e-12, verbosity=1, do_gauge_fixing=true));
    push!(subspace_ALs, AL1)
    push!(subspace_errs, AL1 - AL)

    if maintain_subspace_size
        popfirst!(subspace_ALs)
        popfirst!(subspace_errs)
    end

    return AL1, AR1, AC1, C1, power_method_conv
end

M = 10
max_diis_step = 5
ΔM = 5
tol = 1e-12
damping_factor = 1e-8

for ix in 1:M
    AL, AR, AC, C, power_method_conv = power_iteration_step!(subspace_ALs, subspace_errs, T, AL, AR, AC, C; maintain_subspace_size=false)
    println("Power method convergence: $power_method_conv")
    (power_method_conv < tol) && break
end

if max_diis_step > 0
    B = VUMPSAutoDiff.initialize_ovlpmat(subspace_errs; damping_factor=damping_factor, inner=inner)
    for diis_step in 1:max_diis_step
        Anew = VUMPSAutoDiff.DIIS_extrapolation(B, subspace_ALs)
        println("DIIS extrapolation step $diis_step")
        
        AL, R = VUMPSAutoDiff.left_canonical_QR(Anew; verbosity=1);
        AR, L = VUMPSAutoDiff.right_canonical_QR(Anew; verbosity=1);
        C = R * L;
        AC = AL * C;

        for _ in 1:ΔM
            AL, AR, AC, C, power_method_conv = power_iteration_step!(subspace_ALs, subspace_errs, T, AL, AR, AC, C; maintain_subspace_size=true)
            println("Power method convergence: $power_method_conv")
            (power_method_conv < tol) && break#return (AL, AR, AC, C)
        end
        VUMPSAutoDiff.update_ovlpmat!(B, ΔM, subspace_errs; damping_factor=damping_factor, inner=inner)
    end
end



for ix in 1:100
    AL, AR, AC, C, power_method_conv = power_step(T, AL, AR, AC, C)
    println("Power method convergence: $power_method_conv")
end




    O = tensor_square_ising_O(asinh(1) / 2 / 2)
    
    function _F1(T)
        AL1, AR1 = vumps(T; A=A, verbosity=0)
        TM = MPSMPOMPSTransferMatrix(AL1, T, AL1)
        EL = left_env(TM)
        ER = right_env(TM)

        @tensor a = EL[4; 1 2] * AL1[1 3; 6] * O[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
        @tensor b = EL[4; 1 2] * AL1[1 3; 6] * T[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
        return real(a/b)
    end