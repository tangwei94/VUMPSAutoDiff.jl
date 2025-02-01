
T = tensor_square_ising(asinh(1) / 2);
A = rand(ComplexF64, ℂ^2*ℂ^2, ℂ^2);
AL, R = VUMPSAutoDiff.left_canonical_QR(A; enable_warning=true);
AR, L = VUMPSAutoDiff.right_canonical_QR(A; enable_warning=true);
C = R * L;

AC = AL * C;
@tensor AC1[-1 -2; -3] := AR[1 -2; -3] * C[-1; 1];
@show norm(AC - AC1)

function power_step(T, AL, AR, AC, C)
    AL1, AR1 = deepcopy(AL), deepcopy(AR)
    return vomps!(AL1, AR1, T, AL, AR, AC, C, VOMPSOptions(maxiter=100, tol=1e-12, verbosity=1, do_gauge_fixing=true));
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