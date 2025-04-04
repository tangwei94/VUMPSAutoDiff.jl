# some other tests using pulling through condition. 

# TODO. make a test for the case where the MPO tensor has local C4 symmetry
# use this to test the error measure and the expansion method

T = tensor_square_ising(asinh(1) / 2)
A = rand(ComplexF64, ℂ^4*ℂ^2, ℂ^4)

options = VOMPSVUMPSComboOptions(; M=3, VUMPS_criterion=1e-4, tol=1e-9, maxiter=100)
AL, AR, AC, C = VUMPSAutoDiff.full_canonicalization(A)
AL, AR, AC, C, vumps_conv0, total_num_iter0 = vumps_vomps_combo_iterations(T, AL, AR, AC, C, options);

TM = MPSMPOMPSTransferMatrix(AL, T, AL); 
EL = rand(ComplexF64, VUMPSAutoDiff.left_space(TM));
λ = VUMPSAutoDiff.left_env!(EL, TM)
left_transfer(TM, EL) - λ * EL |> norm

TM = MPSMPOMPSTransferMatrix(AR, T, AR); 
ER = rand(ComplexF64, VUMPSAutoDiff.right_space(TM));
λ = VUMPSAutoDiff.right_env!(ER, TM)
right_transfer(TM, ER) - λ * ER |> norm

B2 = two_site_variation(AL, AR, C, EL, ER, T)

norm(B2)
tsvd(B2)
AL, AR, C, AC = changebonds!(AL, AR, C, B2, truncerr(1e-4));
convert(Array, AL)[:, 1, :]
convert(Array, AL)[:, 2, :]
convert(Array, AR)[:, 1, :]
convert(Array, AR)[:, 2, :]

AL, AR, AC, C, vumps_conv0, total_num_iter0 = vumps_vomps_combo_iterations(T, AL, AR, AC, C, options);
TM = MPSMPOMPSTransferMatrix(AL, T, AL); 
EL = rand(ComplexF64, VUMPSAutoDiff.left_space(TM));
λ = VUMPSAutoDiff.left_env!(EL, TM)
left_transfer(TM, EL) - λ * EL |> norm

TM = MPSMPOMPSTransferMatrix(AR, T, AR); 
ER = rand(ComplexF64, VUMPSAutoDiff.right_space(TM));
λ = VUMPSAutoDiff.right_env!(ER, TM)
right_transfer(TM, ER) - λ * ER |> norm
B2 = two_site_variation(AL, AR, C, EL, ER, T)

norm(B2)
AL, AR, C, AC = changebonds!(AL, AR, C, B2, truncerr(1e-4));

convert(Array, AL)[:, 1, :]
convert(Array, AL)[:, 2, :]
convert(Array, AR)[:, 1, :]
convert(Array, AR)[:, 2, :]

AL, AR, AC, C, vumps_conv0, total_num_iter0 = vumps_vomps_combo_iterations(T, AL, AR, AC, C, options);
TM = MPSMPOMPSTransferMatrix(AL, T, AL); 
EL = rand(ComplexF64, VUMPSAutoDiff.left_space(TM));
λ = VUMPSAutoDiff.left_env!(EL, TM)
left_transfer(TM, EL) - λ * EL |> norm

TM = MPSMPOMPSTransferMatrix(AR, T, AR); 
ER = rand(ComplexF64, VUMPSAutoDiff.right_space(TM));
λ = VUMPSAutoDiff.right_env!(ER, TM)
right_transfer(TM, ER) - λ * ER |> norm
B2 = two_site_variation(AL, AR, C, EL, ER, T)

norm(B2)
AL, AR, C, AC = changebonds!(AL, AR, C, B2, truncerr(1e-4));

AL, AR, AC, C, vumps_conv0, total_num_iter0 = vumps_vomps_combo_iterations(T, AL, AR, AC, C, options);
TM = MPSMPOMPSTransferMatrix(AL, T, AL); 
EL = rand(ComplexF64, VUMPSAutoDiff.left_space(TM));
λ = VUMPSAutoDiff.left_env!(EL, TM)
left_transfer(TM, EL) - λ * EL |> norm

TM = MPSMPOMPSTransferMatrix(AR, T, AR); 
ER = rand(ComplexF64, VUMPSAutoDiff.right_space(TM));
λ = VUMPSAutoDiff.right_env!(ER, TM)
right_transfer(TM, ER) - λ * ER |> norm
B2 = two_site_variation(AL, AR, C, EL, ER, T)

norm(B2)
AL, AR, C, AC = changebonds!(AL, AR, C, B2, truncerr(1e-4));

AL, AR, AC, C, vumps_conv0, total_num_iter0 = vumps_vomps_combo_iterations(T, AL, AR, AC, C, options);
TM = MPSMPOMPSTransferMatrix(AL, T, AL); 
EL = rand(ComplexF64, VUMPSAutoDiff.left_space(TM));
λ = VUMPSAutoDiff.left_env!(EL, TM)
left_transfer(TM, EL) - λ * EL |> norm

TM = MPSMPOMPSTransferMatrix(AR, T, AR); 
ER = rand(ComplexF64, VUMPSAutoDiff.right_space(TM));
λ = VUMPSAutoDiff.right_env!(ER, TM)
right_transfer(TM, ER) - λ * ER |> norm
B2 = two_site_variation(AL, AR, C, EL, ER, T)

norm(B2)
AL, AR, C, AC = changebonds!(AL, AR, C, B2, truncerr(1e-4));
# B2 goes below 1e-4, requirement satisfied. 



D = isomorphism(ℂ^2, (ℂ^2)')
@tensor B[-1 -2; -3] := EL[-1; -3 1] * D[-2; 1]

BL, X = VUMPSAutoDiff.left_canonical_QR(B)
U, conv_meas = gauge_fixing(AL, BL);

AR, C = VUMPSAutoDiff.right_canonical_QR(AL)
_, SC, _ = tsvd(C)
SC

@tensor BL1[-1 -2; -3] := BL[1; -2 2] * (U)[-1; 1] * (U')[2; -3]
β = BL1[1] /AL[1]
norm(β)
BL1 - AL * β |> norm

@tensor B1[-1 -2; -3] := B[1; -2 2] * X[-1; 1] * inv(X)[2; -3]
α = (BL)[1] / B1[1]
BL - B1 * α |> norm

_, SUX, _ = tsvd((U * X))
SC
SUX

corner = (U*X)
corner = corner / corner[1]

Scorner, _ = eigen(corner)

Scorner = Scorner / Scorner[1]

a = convert(Array, Scorner ^2 - SC)
using LinearAlgebra
diag(a)

SC = SC / SC[1]

(Scorner[1] / SC2[1]) * SC2 - Scorner