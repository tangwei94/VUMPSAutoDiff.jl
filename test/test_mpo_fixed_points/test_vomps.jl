# only a demo for mpo_power_iterations using MPOPowerIterationOptions
@testset "MPO Power Iterations" for ix in 1:3
    T = tensor_square_ising(asinh(1) / 2)
    B = rand(ComplexF64, ℂ^8*ℂ^2, ℂ^8)

    options = MPOPowerIterationOptions(; diis_criterion=1e-4, max_diis_step=10)
    BL, BR, BC, C = VUMPSAutoDiff.full_canonicalization(B)
    BL1, BR1, BC1, C1, power_method_conv1, _ = mpo_power_iterations(T, BL, BR, BC, C, options);

    nodiis_options = MPOPowerIterationOptions(; diis_criterion=1e-8, M=0, max_diis_step=0)
    BL, BR, BC, C = VUMPSAutoDiff.full_canonicalization(B);
    BL2, BR2, BC2, C2, power_method_conv2, _ = mpo_power_iterations(T, BL, BR, BC, C, nodiis_options);

    @test power_method_conv2 < power_method_conv1
end

@testset "VOMPS+VUMPS" for ix in 1:3
    T = tensor_square_ising(asinh(1) / 2)
    B = rand(ComplexF64, ℂ^8*ℂ^2, ℂ^8)
    # VOMPS+VUMPS
    options = VOMPSVUMPSComboOptions(; M=3, VUMPS_criterion=1e-4, tol=1e-9, maxiter=100)
    BL, BR, BC, C = VUMPSAutoDiff.full_canonicalization(B)
    _, _, _, _, vumps_conv0, total_num_iter0 = vumps_vomps_combo_iterations(T, BL, BR, BC, C, options);

    # pure VOMPS, force it to reach 1e-9 convergence criterion. check VUMPS convergence afterwards.
    options1 = VOMPSVUMPSComboOptions(; M=0, VUMPS_criterion=1e-9, tol=1e-9, maxiter=1)
    BL, BR, BC, C = VUMPSAutoDiff.full_canonicalization(B)
    _, _, _, _, vumps_conv1, total_num_iter1 = vumps_vomps_combo_iterations(T, BL, BR, BC, C, options1);
    @show vumps_conv1, total_num_iter1

    # pure VUMPS after VOMPS reaches 1e-4 convergence criterion
    options2 = VOMPSVUMPSComboOptions(; M=0, VUMPS_criterion=1e-4, tol=1e-9, maxiter=100)
    BL, BR, BC, C = VUMPSAutoDiff.full_canonicalization(B)
    _, _, _, _, vumps_conv2, total_num_iter2 = vumps_vomps_combo_iterations(T, BL, BR, BC, C, options2);
    @show vumps_conv2, total_num_iter2

    # for this specific example we tested, both VOMPS+VUMPS and pure VUMPS should converge in less than 1500 iterations, and they should both outperform the pure power method
    @show vumps_conv0, vumps_conv1, vumps_conv2
    @test vumps_conv0 < 1e-9
    @test vumps_conv2 < 1e-9
    @test vumps_conv2 < vumps_conv1
    @test vumps_conv0 < vumps_conv1

    @show total_num_iter0, total_num_iter1, total_num_iter2
    @test total_num_iter0 < 1500
    @test total_num_iter2 < 1500
    @test total_num_iter0 < total_num_iter1
    @test total_num_iter2 < total_num_iter1
end