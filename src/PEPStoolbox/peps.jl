function delta_fusing_virtual_bonds(A::PEPSTensor)
    V = conj(space(A, 3)) # !
    δ = isomorphism(V*V', fuse(V, V')) # !
    return permute(δ, ((1, ), (2, 3)))
end
@non_differentiable delta_fusing_virtual_bonds(A::PEPSTensor)

function link_tensor_on_PEPS_bond(V::ElementarySpace) 
    D = Tensor(zeros, ComplexF64, V*V)
    for (f1, f2) in fusiontrees(D)
        Ddiag = view(D[f1, f2], diagind(D[f1, f2]))
        Ddiag .= 1.0
    end
    return D 
end
@non_differentiable link_tensor_on_PEPS_bond(V::ElementarySpace)

function double_layer_tensor_on_site_term(opU::OnSiteTerm, A::PEPSTensor)
    δ = delta_fusing_virtual_bonds(A)
    @tensor T_opU[-1 -2 -3 -4] := A[1 11; 3 5 7 9] * opU[2; 1] * conj(A[2 11; 4 6 8 10]) * δ[3; 4 -1] * δ[5; 6 -2] * δ[7; 8 -3] * δ[9; 10 -4]
    return permute(T_opU, ((), (1, 2, 3, 4)))
end

function double_layer_tensor_bond_term(opL::BondTermLeft, opR::BondTermRight, A::PEPSTensor)
    δ = delta_fusing_virtual_bonds(A)

    @tensor T_opL[-1 -2 -3 -4 -5] := A[1 11; 3 5 7 9] * opL[2; 1 -5] * conj(A[2 11; 4 6 8 10]) * δ[3; 4 -1] * δ[5; 6 -2] * δ[7; 8 -3] * δ[9; 10 -4]
    @tensor T_opR[-1; -2 -3 -4 -5] := A[1 11; 3 5 7 9] * opR[-1 2; 1] * conj(A[2 11; 4 6 8 10]) * δ[3; 4 -2] * δ[5; 6 -3] * δ[7; 8 -4] * δ[9; 10 -5]
    return permute(T_opL, ((), (1, 2, 3, 4, 5))), T_opR
end

function double_layer_tensor_PEPS_site(A::PEPSTensor)
    opI = ignore_derivatives() do
        id(space(A, 1))
    end
    return double_layer_tensor_on_site_term(opI, A)
end

function double_layer_tensor_PEPS_bond(D::PEPSBondTensor)
    V = space(D, 1)
    δ = isomorphism(fuse(V, V'), V*V') # !
    δ = permute(δ, ((1, 3), (2, )))

    @tensor TD[-1 -2] := δ[-1 2; 1] * D[1 3] * conj(D[2 4]) * δ[-2 4; 3]
    return TD
end
@non_differentiable double_layer_tensor_PEPS_bond(D::PEPSBondTensor)

function build_MPO(A::PEPSTensor)
    V = space(A, 3)'
    D = link_tensor_on_PEPS_bond(V)
    TA = double_layer_tensor_PEPS_site(A);
    TD = double_layer_tensor_PEPS_bond(D);
    @tensor Tfull[-1 -2; -3 -4] := TA[2 1 -3 -4] * TD[-1 1] * TD[-2 2];
    return Tfull
end

