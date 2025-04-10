using JLD2
M = load("M.jld2", "M")
M1 = permutedims(M, (1, 2, 4, 3))
M2 = permutedims(conj.(M), (1, 4, 2, 3))

T = TensorMap(M1, ℂ^4*ℂ^4, ℂ^4*ℂ^4)
Tdag = TensorMap(M2, ℂ^4*ℂ^4, ℂ^4*ℂ^4)
χ = 4
A = rand(ComplexF64, ℂ^χ*ℂ^4, ℂ^χ)

T.data

options2 = VOMPSVUMPSComboOptions(; M=0, VUMPS_criterion=1e-4, tol=1e-9, maxiter=1000)
AL_b, AR_b, AC_b, C_b = VUMPSAutoDiff.full_canonicalization(A)

space(AL_b)
AL_b, AR_b, AC_b, C_b, vumps_conv2_b, total_num_iter2_b = vumps_vomps_combo_iterations(T, AL_b, AR_b, AC_b, C_b, options2);

EL_b = left_env(MPSMPOMPSTransferMatrix(AL_b, T, AL_b))
ER_b = right_env(MPSMPOMPSTransferMatrix(AR_b, T, AR_b))

B2 = two_site_variation(AL_b, AR_b, C_b, EL_b, ER_b, T) 
norm(B2)

AL_b, AR_b, C_b, AC_b = changebonds!(AL_b, AR_b, C_b, B2, truncerr(1e-4));



