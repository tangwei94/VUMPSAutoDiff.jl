# only a demo for mpo_power_iterations using MPOPowerIterationOptions
T = tensor_square_ising(asinh(1) / 2)
A = rand(ComplexF64, ℂ^4*ℂ^2, ℂ^4) 

AL1, AR1 = vumps(T; A=A, verbosity=2);

B = rand(ComplexF64, ℂ^8*ℂ^2, ℂ^8)
BL, BR, BC, C = VUMPSAutoDiff.full_canonicalization(B)

options = MPOPowerIterationOptions(; diis_criterion=1e-4, max_diis_step=20)
BL1, BR1, BC1, C1 = mpo_power_iterations(T, BL, BR, BC, C, options);

nodiis_options = MPOPowerIterationOptions(; M=1000, max_diis_step=0)
BL, BR, BC, C = VUMPSAutoDiff.full_canonicalization(B);
BL2, BR2, BC2, C2 = mpo_power_iterations(T, BL, BR, BC, C, nodiis_options);



B = rand(ComplexF64, ℂ^8*ℂ^2, ℂ^8)

# VOMPS+VUMPS
options = VOMPSVUMPSComboOptions(; M=10, VUMPS_criterion=1e-4, tol=1e-9, maxiter=100)
BL, BR, BC, C = VUMPSAutoDiff.full_canonicalization(B)
vumps_vomps_combo_iterations(T, BL, BR, BC, C, options);

# pure VOMPS, force it to reach 1e-9 convergence criterion. check VUMPS convergence afterwards.
options1 = VOMPSVUMPSComboOptions(; M=10, VUMPS_criterion=1e-9, tol=1e-9, maxiter=1)
BL, BR, BC, C = VUMPSAutoDiff.full_canonicalization(B)
vumps_vomps_combo_iterations(T, BL, BR, BC, C, options1);

# pure VUMPS after VOMPS reaches 1e-4 convergence criterion
options2 = VOMPSVUMPSComboOptions(; M=0, VUMPS_criterion=1e-4, tol=1e-9, maxiter=100)
BL, BR, BC, C = VUMPSAutoDiff.full_canonicalization(B)
vumps_vomps_combo_iterations(T, BL, BR, BC, C, options2);


