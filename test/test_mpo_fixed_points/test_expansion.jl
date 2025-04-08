@testset "bond dimension expansion" begin
    tol = 1e-4

    T = tensor_square_ising(asinh(1) / 2)
    A = rand(ComplexF64, ℂ^4*ℂ^2, ℂ^4)

    options = VOMPSVUMPSComboOptions(; M=3, VUMPS_criterion=1e-4, tol=1e-9, maxiter=100)
    AL, AR, AC, C = VUMPSAutoDiff.full_canonicalization(A)
    AL, AR, AC, C, vumps_conv0, total_num_iter0 = vumps_vomps_combo_iterations(T, AL, AR, AC, C, options);

    function get_and_test_environments(AL, AR, T)
        # obtain left and right environments EL and ER
        TM = MPSMPOMPSTransferMatrix(AL, T, AL); 
        EL = rand(ComplexF64, VUMPSAutoDiff.left_space(TM));
        λ = VUMPSAutoDiff.left_env!(EL, TM)
        @test norm(left_transfer(TM, EL) - λ * EL) < 1e-9

        TM = MPSMPOMPSTransferMatrix(AR, T, AR); 
        ER = rand(ComplexF64, VUMPSAutoDiff.right_space(TM));
        λ = VUMPSAutoDiff.right_env!(ER, TM)
        @test norm(right_transfer(TM, ER) - λ * ER) < 1e-9
        return EL, ER
    end

    err = Inf
    total_number_of_iterations = 0
    for _ in 1:10
        EL, ER = get_and_test_environments(AL, AR, T)

        # obtain two-site variation, which serves as an error measure and provides information for bond dimension expansion
        B2 = two_site_variation(AL, AR, C, EL, ER, T)
        err = norm(B2)
        @show space(AL), err
        if err > tol
            # use the information in B2 to increase the bond dimension
            AL, AR, C, AC = changebonds!(AL, AR, C, B2, truncerr(tol));
            AL, AR, AC, C, vumps_conv0, total_num_iter0 = vumps_vomps_combo_iterations(T, AL, AR, AC, C, options);
            total_number_of_iterations = total_num_iter0
        else
            break
        end
    end
    @test err < tol

    randomize!(AL)
    AL, AR, AC, C = VUMPSAutoDiff.full_canonicalization(AL)
    AL, AR, AC, C, vumps_conv0, total_num_iter_random_init = vumps_vomps_combo_iterations(T, AL, AR, AC, C, options);

    EL, ER = get_and_test_environments(AL, AR, T)
    B2 = two_site_variation(AL, AR, C, EL, ER, T)
    @test norm(B2) < 1e-4
    @test total_num_iter_random_init > total_number_of_iterations

end
