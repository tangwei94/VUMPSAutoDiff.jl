using TensorKit, KrylovKit
using Zygote, ChainRulesCore, ChainRules
using VUMPSAutoDiff

function δ_ising()
    δ = zeros(ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2)
    δ[1, 1, 1, 1] = 1
    δ[2, 2, 2, 2] = 1 
    return δ
end
@non_differentiable δ_ising()
function b_ising(β)
    b2 = TensorMap(ComplexF64[exp(β) exp(-β); exp(-β) exp(β)], ℂ^2, ℂ^2)
    return sqrt(b2)
end
# The backward rule for the sqrt of TensorMap is not implemented. manually implement it :(
function ChainRulesCore.rrule(::typeof(b_ising), β)
    b = b_ising(β)
    
    function b_pushback(_∂b)
        ∂b = unthunk(_∂b)

        d11 = (1/2) * (cosh(β)/(sqrt(2)*sqrt(sinh(β))) + sinh(β)/(sqrt(2)*sqrt(cosh(β))))
        d12 = (1/2) * (-cosh(β)/(sqrt(2)*sqrt(sinh(β))) + sinh(β)/(sqrt(csch(β)*sinh(2*β))))
        d21 = (1/2) * (-cosh(β)/(sqrt(2)*sqrt(sinh(β))) + sinh(β)/(sqrt(csch(β)*sinh(2*β))))
        d22 = (1/2) * (cosh(β)/(sqrt(2)*sqrt(sinh(β))) + sinh(β)/(sqrt(2)*sqrt(cosh(β))))
    
        dbdβ = TensorMap(ComplexF64[d11 d12; d21 d22], ℂ^2, ℂ^2)
        return NoTangent(), tr(∂b * dbdβ')
    end
    return b, b_pushback 
end

# defines the MPO tensor for the partition function of the classical Ising model at temperature β  
function tensor_square_ising(β)
    b = b_ising(β)
    δ = δ_ising()
    @tensor T[-1 -2 ; -3 -4] := b[-1; 1] * b[-2; 2] * b[3; -3] * b[4; -4] * δ[1 2; 3 4]
    return T
end

β = asinh(1.0) / 2

# forward calculation that computes the free energy of the classical Ising model at temperature β
function _F1(β::Float64)
    T = tensor_square_ising(β)
    A = ignore_derivatives() do
        rand(ComplexF64, ℂ^4*ℂ^2, ℂ^4) 
    end
    AL1, AR1 = vumps(T; A=A, verbosity=2)
    TM = MPSMPOMPSTransferMatrix(AL1, T, AL1)
    EL = left_env(TM)
    ER = right_env(TM)

    @tensor a = EL[4; 1 2] * AL1[1 3; 6] * T[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
    @tensor b = EL[3; 1 2] * ER[1 2; 3]
    return - log(real(a/b)) / β
end

# compute the gradient of the free energy with respect to β using AD
y, grad_y = withgradient(_F1, β)
dβ = 1e-4

# verify the AD gradient by a finite difference method
grad_y_finitediff = (_F1(β + dβ) - _F1(β - dβ)) / (2*dβ)
@show (grad_y[1] - grad_y_finitediff) < 1e-4
