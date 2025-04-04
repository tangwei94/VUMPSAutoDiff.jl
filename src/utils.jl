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

# copied from https://github.com/QuantumKitHub/MPSKit.jl/blob/d30ef9e97dec9375b43574c9877820e8922574f0/src/utility/utility.jl#L20-L21
_firstspace(t::AbstractTensorMap) = space(t, 1)
_lastspace(t::AbstractTensorMap) = space(t, numind(t))


# copied from MPSKit.jl: https://github.com/QuantumKitHub/MPSKit.jl/blob/d30ef9e97dec9375b43574c9877820e8922574f0/src/utility/utility.jl#L1-L6
function _transpose_front(t::AbstractTensorMap) # make TensorMap{S,N₁+N₂-1,1}
    return repartition(t, numind(t) - 1, 1)
end
function _transpose_tail(t::AbstractTensorMap) # make TensorMap{S,1,N₁+N₂-1}
    return repartition(t, 1, numind(t) - 1)
end



#VectorInterface.scalartype(x::ZeroTangent) = Float64

#TensorKitChainRulesCoreExt = Base.get_extension(TensorKit, :TensorKitChainRulesCoreExt)