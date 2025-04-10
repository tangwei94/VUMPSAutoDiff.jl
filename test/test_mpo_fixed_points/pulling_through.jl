# some other tests using pulling through condition. 

# temporary
left_canonical_QR = VUMPSAutoDiff.left_canonical_QR
right_canonical_QR = VUMPSAutoDiff.right_canonical_QR
mps_fidelity = VUMPSAutoDiff.mps_fidelity

χ = 4
d = 4
Tmat = rand(Float64, d, d, d, d) # d=2 seems trivial???
Tmat = 0.5 * (Tmat + permutedims(Tmat, (2, 3, 4, 1)) + permutedims(Tmat, (3, 4, 1, 2)) + permutedims(Tmat, (4, 1, 2, 3)))
Tmat = Tmat / norm(Tmat)
Tmat - permutedims(Tmat, (1, 4, 3, 2)) |> norm  # does not have reflection symmetry. 
T = TensorMap(permutedims(Tmat, (1, 2, 4, 3)), ℂ^χ ⊗ ℂ^d, ℂ^d ⊗ ℂ^χ)

# for Ising model, check the fixed point tensor satisfies the pulling through condition.
#T = tensor_square_ising(asinh(1) / 2)

δ = permute(isomorphism((ℂ^χ), (ℂ^χ)'), ((2, 1), ()))

A = rand(ComplexF64, ℂ^χ*ℂ^d, ℂ^χ)

options = VOMPSVUMPSComboOptions(; M=3, VUMPS_criterion=1e-4, tol=1e-9, maxiter=100)
AL, AR, AC, C = VUMPSAutoDiff.full_canonicalization(A)
AL, AR, AC, C, vumps_conv0, total_num_iter0 = vumps_vomps_combo_iterations(T, AL, AR, AC, C, options);

TM = MPSMPOMPSTransferMatrix(AL, T, AL); 
EL = rand(ComplexF64, VUMPSAutoDiff.left_space(TM));
λ = VUMPSAutoDiff.left_env!(EL, TM)
left_transfer(TM, EL) - λ * EL |> norm # TODO. add this to a specific test.

@tensor B[-1 -2; -3] := EL[-1; -3 2] * δ[-2; 2]
BR, L = right_canonical_QR(B)
mps_fidelity(B, BR) # TODO. add this to a specific test.

@tensor B1[-1 -2; -3] := B[1 -2; 2] * inv(L)[-1; 1] * L[2; -3]
B1[1]/BR[1] * BR - B1 |> norm  # TODO. add this to a specific test

# B -> BR, then A should follow the same gauge transformation as B
@tensor A[-1 -2; -3] = AL[1 -2; 2] * inv(L)[-1; 1] * L[2; -3]

@tensor Trot[-1 -2; -3 -4] := T[1 -1; -4 2] * δ[2 -2] * conj(δ[-3 1])
Trot - T |> norm # TODO. wrap this as a test. C4 rotational invariance

ER_B = rand(ComplexF64, VUMPSAutoDiff.right_space(TM_B));
λB = VUMPSAutoDiff.right_env!(ER_B, TM_B)
λ - λB 

TM_B = MPSMPOMPSTransferMatrix(BR, T, BR);
right_transfer(TM_B, A) - λ * A |> norm