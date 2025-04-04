using JLD2
M = load("M.jld2", "M")
M1 = permutedims(M, (1, 2, 4, 3))
M2 = permutedims(conj.(M), (1, 4, 2, 3))

T = TensorMap(M1, ℂ^4*ℂ^4, ℂ^4*ℂ^4)
Tdag = TensorMap(M2, ℂ^4*ℂ^4, ℂ^4*ℂ^4)
χ = 12
B = rand(ComplexF64, ℂ^χ*ℂ^4, ℂ^χ)

T.data

options2 = VOMPSVUMPSComboOptions(; M=0, VUMPS_criterion=1e-4, tol=1e-9, maxiter=1000)
BL, BR, BC, C = VUMPSAutoDiff.full_canonicalization(B)
AL, AR, AC, C = VUMPSAutoDiff.full_canonicalization(B)
BL_b, BR_b, BC_b, C_b, vumps_conv2_b, total_num_iter2_b = vumps_vomps_combo_iterations(T, BL, BR, BC, C, options2);
BL_t, BR_t, BC_t, C_t, vumps_conv2_t, total_num_iter2_t = vumps_vomps_combo_iterations(Tdag, BL, BR, BC, C, options2);
  
power_method_conv = Inf
total_num_iter = 0
while power_method_conv > options2.VUMPS_criterion && total_num_iter < 1e4
    AL1, AR1 = deepcopy(AL), deepcopy(AR)
    AL2, AR2 = deepcopy(AL), deepcopy(AR)
    AL1, AR1, AC1, C1, power_method_conv, num_iter = vomps!(AL1, AR1, T, AL, AR, AC, C, VOMPSOptions(maxiter=250, tol=1e-12, verbosity=1, do_gauge_fixing=true));
    AL2, AR2, AC2, C2, power_method_conv, num_iter = vomps!(AL2, AR2, T, AL1, AR1, AC1, C1, VOMPSOptions(maxiter=250, tol=1e-12, verbosity=1, do_gauge_fixing=true));
    power_method_conv = VUMPSAutoDiff.gauge_fix!(AL2, AR2, AC2, C2, AL)
    AL, AR, AC, C = AL2, AR2, AC2, C2

    total_num_iter += num_iter
    println("Power method convergence: $power_method_conv, total_num_iter: $total_num_iter")
    @show norm(AL), norm(AR), norm(AC), norm(C)
end

VUMPSAutoDiff.mps_fidelity(BL_b, BL_t)

