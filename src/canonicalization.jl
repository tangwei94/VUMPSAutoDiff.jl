function right_canonical_QR_operation!(L::MPSBondTensor, AR::MPSTensor, A::MPSTensor)
    L1, Q = rightorth(A * L, ((1, ), (2, 3)))
    permute!(AR, Q, ((1, 2), (3, )))
    rmul!(L1, 1/norm(L1))
    α = overall_u1_phase(L, L1)
    δ = norm(L - L1 * α)
    copy!(L, L1)
    return δ
end

function right_canonical_QR(A::MPSTensor; tol::Float64=1e-12, maxiter::Int=200, verbosity::Int=0)

    #lop = MPSMPSTransferMatrix(A, A)
    #ρR = right_env(lop)
    #U, S, _ = tsvd(ρR)
    #L = U * sqrt(S)

    L = id(ComplexF64, space(A, 1)) / dim(space(A, 1))
    AR = similar(A, space(A))
    δ = right_canonical_QR_operation!(L, AR, A)

    for ix in 1:maxiter
        if ix % 10 == 0 
            lop = MPSMPSTransferMatrix(A, AR)
            copy!(L, right_env(lop)') # TODO. tol = max(tol, δ/10)
        end
        δ = right_canonical_QR_operation!(L, AR, A)
        (verbosity >= 2) && println("right_canonical_QR: step $ix: δ = $δ")
        (δ < tol) && break 
    end

    (verbosity >= 1) && println("right_canonical_QR: final convergence: δ = $δ")

    return AR, L # originally here we return L0'. don't remember why.
end

function left_canonical_QR_operation!(R::MPSBondTensor, AL::MPSTensor, A::MPSTensor)
    @tensor AL[-1 -2; -3] = R[-1; 1] * A[1 -2; -3]
    Q, R1 = leftorth!(AL)
    copy!(AL, Q)
    rmul!(R1, 1/norm(R1))
    α = overall_u1_phase(R, R1)
    δ = norm(R - R1 * α)
    copy!(R, R1)

    return δ
end

function left_canonical_QR(A::TensorMap{T, ComplexSpace, 2, 1}; tol::Float64=1e-12, maxiter::Int=200, verbosity::Int=0) where T

    #lop = MPSMPSTransferMatrix(A, A)
    #ρL = left_env(lop)
    #_, S, V = tsvd(ρL)
    #R = sqrt(S) * V
    
    R = id(ComplexF64, space(A, 1)) / dim(space(A, 1))
    AL = similar(A, space(A))
    δ = left_canonical_QR_operation!(R, AL, A)

    for ix in 1:maxiter
        if ix % 10 == 0 
            lop = MPSMPSTransferMatrix(A, AL) # TODO. tol = max(tol, δ/10)
            copy!(R, left_env(lop)')
        end
        δ = left_canonical_QR_operation!(R, AL, A)
        (verbosity >= 2) && println("left_canonical_QR: step $ix: δ = $δ")
        (δ < tol) && break 
    end

    (verbosity >= 1) && println("left_canonical_QR: final convergence: δ = $δ")
    return AL, R
end