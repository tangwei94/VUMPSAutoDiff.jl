const RhoTensor  = AbstractTensorMap{T,S,1,1} where {T,S}
const EnvTensorL = AbstractTensorMap{T,S,1,2} where {T,S}
const EnvTensorR = AbstractTensorMap{T,S,2,1} where {T,S}
const MPSTensor = AbstractTensorMap{T,S,2,1} where {T,S}
const MPSBondTensor = AbstractTensorMap{T,S,1,1} where {T,S}
const MPOTensor = AbstractTensorMap{T,S,2,2} where {T,S}

# PEPS Tensor: physical space * auxiliary space <- virtual space^4. Only support PEPS on the square lattice for now.
const PEPSTensor = AbstractTensorMap{T,S,2,4} where {T,S}
const PEPSBondTensor = AbstractTensorMap{T,S,1,1} where {T,S}

const OnSiteTerm = AbstractTensorMap{T,S,1,1} where {T,S}
const BondTermLeft = AbstractTensorMap{T,S,1,2} where {T,S}
const BondTermRight = AbstractTensorMap{T,S,2,1} where {T,S}

# copied from MPSKit.jl
function fill_data!(a::TensorMap, dfun)
    for (k, v) in blocks(a)
        map!(x -> dfun(typeof(x)), v, v)
    end

    return a
end
randomize!(a::TensorMap) = fill_data!(a, randn)

#VectorInterface.scalartype(x::ZeroTangent) = Float64

#TensorKitChainRulesCoreExt = Base.get_extension(TensorKit, :TensorKitChainRulesCoreExt)