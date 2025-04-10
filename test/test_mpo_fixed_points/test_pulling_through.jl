# some tests for pulling through condition. 

# temporary
left_canonical_QR = VUMPSAutoDiff.left_canonical_QR
right_canonical_QR = VUMPSAutoDiff.right_canonical_QR
mps_fidelity = VUMPSAutoDiff.mps_fidelity

@testset "Pulling through condition" for ix  in 1:5
    function check_pulling_through(T, χ, d)
        δ = permute(isomorphism((ℂ^d), (ℂ^d)'), ((2, 1), ()))
        A = rand(ComplexF64, ℂ^χ*ℂ^d, ℂ^χ)

        options = VOMPSVUMPSComboOptions(; M=3, VUMPS_criterion=1e-4, tol=1e-9, maxiter=100)
        AL, AR, AC, C = VUMPSAutoDiff.full_canonicalization(A)
        AL, AR, AC, C, vumps_conv0, total_num_iter0 = vumps_vomps_combo_iterations(T, AL, AR, AC, C, options);

        TM = MPSMPOMPSTransferMatrix(AL, T, AL); 
        EL = rand(ComplexF64, VUMPSAutoDiff.left_space(TM));
        λ = VUMPSAutoDiff.left_env!(EL, TM)
        @test norm(left_transfer(TM, EL) - λ * EL) < sqrt(eps(1.0)) # TODO. add this to a specific test for MPSMPOMPSTransferMatrix

        @tensor B[-1 -2; -3] := EL[-1; -3 2] * δ[-2; 2]
        BR, L = right_canonical_QR(B)
        @test mps_fidelity(B, BR) > 1 - sqrt(eps(1.0)) # TODO. add this to a specific test for canonicalization.

        @tensor B1[-1 -2; -3] := B[1 -2; 2] * inv(L)[-1; 1] * L[2; -3]
        @test norm(B1[1]/BR[1] * BR - B1) < sqrt(eps(1.0)) # TODO. add this to a specific test for canonicalization.

        # B -> BR, then A should follow the same gauge transformation as B
        @tensor A[-1 -2; -3] = AL[1 -2; 2] * inv(L)[-1; 1] * L[2; -3]

        @tensor Trot[-1 -2; -3 -4] := T[1 -1; -4 2] * δ[2 -2] * conj(δ[-3 1])
        @test norm(Trot - T) < sqrt(eps(1.0)) # this confirms the C4 rotational invariance of the MPO tensor. 

        TM_B = MPSMPOMPSTransferMatrix(BR, T, BR);
        ER_B = rand(ComplexF64, VUMPSAutoDiff.right_space(TM_B));
        λB = VUMPSAutoDiff.right_env!(ER_B, TM_B)

        check1 = (abs(λ - λB) < 100*sqrt(eps(norm(λ)))) 
        check2 = (norm(right_transfer(TM_B, A) - λ * A) < 100*sqrt(eps(norm(λ))))
        
        AL1, _ = left_canonical_QR(A)
        AL2, _ = left_canonical_QR(ER_B)
        _, meas = gauge_fixing(AL1, AL2)
        check3 = (meas < sqrt(eps(1.0)))

        return check1, check2, check3
    end

    # test for a random MPO tensor T with C4 rotational symmetry but no reflection symmetry
    χ = 2
    d = 4 # for d = 2, rotational symmetry will always guarantee the reflection symmetry
    T = let
        # Generate random tensor and enforce C4 rotational symmetry
        Tmat = rand(Float64, d, d, d, d)
        Tmat = sum(permutedims(Tmat, circshift(1:4, i)) for i in 0:3) / 4
        Tmat = Tmat / norm(Tmat)  # Normalize
        
        # Keep generating until we find one without reflection symmetry, although this is not necessary in most cases.
        while norm(Tmat - permutedims(Tmat, (1, 4, 3, 2))) < sqrt(eps(1.0))
            Tmat = rand(Float64, d, d, d, d)
            Tmat = sum(permutedims(Tmat, circshift(1:4, i)) for i in 0:3) / 4
            Tmat = Tmat / norm(Tmat)
        end
        
        # Convert to TensorMap with correct tensor product spaces
        TensorMap(permutedims(Tmat, (1, 2, 4, 3)), ℂ^d ⊗ ℂ^d, ℂ^d ⊗ ℂ^d)
    end

    check1, check2, check3 = check_pulling_through(T, χ, d)
    # with only the rotational symmetry, the pulling through convergence condition is not satisfied by the VUMPS fixed point. 
    # check1 can pass, indicating that the VUMPS fixed point tensor A can neverthless approximate the boundary state very well. This is probably due to the fact that the random tensor T that we used here leads to gapped MPO in most cases. I am not sure whether this is still true when the MPO is gapless. 
    # check2, check3 are expected to fail, indicating that the VUMPS fixed point tensor A does not satisfy the pulling through convergence conditions.
    @show check1
    @test ! check2
    @test ! check3
   
    # test with the 
    χ = 4
    d = 2
    T = tensor_square_ising(asinh(1) / 2)
    check1, check2, check3 = check_pulling_through(T, χ, d)
    @test check1
    @test check2
    @test check3
end


