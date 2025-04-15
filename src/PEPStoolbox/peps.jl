"""
    delta_fusing_virtual_bonds(A::PEPSTensor)

Create a delta tensor for fusing virtual bonds in a PEPS tensor.

# Arguments
- `A::PEPSTensor`: The input PEPS tensor

# Returns
- A permuted isomorphism tensor that fuses the virtual bonds of the PEPS tensor
"""
function delta_fusing_virtual_bonds(A::PEPSTensor)
    V = conj(space(A, 3)) # !
    δ = isomorphism(V*V', fuse(V, V')) # !
    return permute(δ, ((1, ), (2, 3)))
end
@non_differentiable delta_fusing_virtual_bonds(A::PEPSTensor)

"""
    link_tensor_on_PEPS_bond(V::ElementarySpace)

Create a link tensor for a PEPS bond with the given elementary space.

# Arguments
- `V::ElementarySpace`: The elementary space for the bond

# Returns
- A diagonal tensor with ones on the diagonal, representing the link between PEPS sites
"""
function link_tensor_on_PEPS_bond(V::ElementarySpace) 
    D = Tensor(zeros, ComplexF64, V*V)
    for (f1, f2) in fusiontrees(D)
        Ddiag = view(D[f1, f2], diagind(D[f1, f2]))
        Ddiag .= 1.0
    end
    return D 
end
@non_differentiable link_tensor_on_PEPS_bond(V::ElementarySpace)

"""
    double_layer_tensor_on_site_term(opU::OnSiteTerm, A::PEPSTensor)

Construct the double-layer tensor for an on-site term in a PEPS.

# Arguments
- `opU::OnSiteTerm`: The on-site operator
- `A::PEPSTensor`: The PEPS tensor

# Returns
- A tensor representing the double-layer contraction of the on-site term with the PEPS tensor
"""
function double_layer_tensor_on_site_term(opU::OnSiteTerm, A::PEPSTensor)
    δ = delta_fusing_virtual_bonds(A)
    @tensor T_opU[-1 -2 -3 -4] := A[1 11; 3 5 7 9] * opU[2; 1] * conj(A[2 11; 4 6 8 10]) * δ[3; 4 -1] * δ[5; 6 -2] * δ[7; 8 -3] * δ[9; 10 -4]
    return permute(T_opU, ((), (1, 2, 3, 4)))
end

"""
    double_layer_tensor_bond_term(opL::BondTermLeft, opR::BondTermRight, A::PEPSTensor)

Construct double-layer tensors for left and right bond terms in a PEPS.

# Arguments
- `opL::BondTermLeft`: The left bond operator
- `opR::BondTermRight`: The right bond operator
- `A::PEPSTensor`: The PEPS tensor

# Returns
- A tuple containing the double-layer tensors for the left and right bond terms
"""
function double_layer_tensor_bond_term(opL::BondTermLeft, opR::BondTermRight, A::PEPSTensor)
    δ = delta_fusing_virtual_bonds(A)

    @tensor T_opL[-1 -2 -3 -4 -5] := A[1 11; 3 5 7 9] * opL[2; 1 -5] * conj(A[2 11; 4 6 8 10]) * δ[3; 4 -1] * δ[5; 6 -2] * δ[7; 8 -3] * δ[9; 10 -4]
    @tensor T_opR[-1; -2 -3 -4 -5] := A[1 11; 3 5 7 9] * opR[-1 2; 1] * conj(A[2 11; 4 6 8 10]) * δ[3; 4 -2] * δ[5; 6 -3] * δ[7; 8 -4] * δ[9; 10 -5]
    return permute(T_opL, ((), (1, 2, 3, 4, 5))), T_opR
end

"""
    double_layer_tensor_PEPS_site(A::PEPSTensor)

Construct the double-layer tensor for a PEPS site using the identity operator.

# Arguments
- `A::PEPSTensor`: The PEPS tensor

# Returns
- A tensor representing the double-layer contraction of the identity operator with the PEPS tensor
"""
function double_layer_tensor_PEPS_site(A::PEPSTensor)
    opI = ignore_derivatives() do
        id(space(A, 1))
    end
    return double_layer_tensor_on_site_term(opI, A)
end

"""
    double_layer_tensor_PEPS_bond(D::PEPSBondTensor)

Construct the double-layer tensor for a PEPS bond.

# Arguments
- `D::PEPSBondTensor`: The PEPS bond tensor

# Returns
- A tensor representing the double-layer contraction of the bond tensor
"""
function double_layer_tensor_PEPS_bond(D::PEPSBondTensor)
    V = space(D, 1)
    δ = isomorphism(fuse(V, V'), V*V') # !
    δ = permute(δ, ((1, 3), (2, )))

    @tensor TD[-1 -2] := δ[-1 2; 1] * D[1 3] * conj(D[2 4]) * δ[-2 4; 3]
    return TD
end
@non_differentiable double_layer_tensor_PEPS_bond(D::PEPSBondTensor)

"""
    build_MPO(A::PEPSTensor)

Build a Matrix Product Operator (MPO) from a PEPS tensor.

# Arguments
- `A::PEPSTensor`: The input PEPS tensor

# Returns
- A tensor representing the MPO constructed from the PEPS tensor and its bond tensors
"""
function build_MPO(A::PEPSTensor)
    V = space(A, 3)'
    D = link_tensor_on_PEPS_bond(V)
    TA = double_layer_tensor_PEPS_site(A);
    TD = double_layer_tensor_PEPS_bond(D);
    @tensor Tfull[-1 -2; -3 -4] := TA[2 1 -3 -4] * TD[-1 1] * TD[-2 2];
    return Tfull
end

