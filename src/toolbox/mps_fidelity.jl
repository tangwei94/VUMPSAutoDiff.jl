function mps_fidelity(AL1::MPSTensor, AL2::MPSTensor)
    tmpEL = rand(ComplexF64, space(AL1, 1), space(AL2, 1))
    
    TM12 = MPSMPSTransferMatrix(AL1, AL2)
    λ12 = left_env!(tmpEL, TM12)

    TM11 = MPSMPSTransferMatrix(AL1, AL1)
    λ11 = left_env!(tmpEL, TM11)

    TM22 = MPSMPSTransferMatrix(AL2, AL2)
    λ22 = left_env!(tmpEL, TM22)
    
    return sqrt(abs(λ12)^2 / (abs(λ11) * abs(λ22)))
end