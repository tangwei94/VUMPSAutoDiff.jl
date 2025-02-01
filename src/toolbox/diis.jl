"""
    DIIS_extrapolation_alg

A struct representing the Direct Inversion in the Iterative Subspace (DIIS) extrapolation algorithm parameters.

# Fields
- `M::Int`: Maximum number of iterations
- `ΔM::Int`: Number of additional iterations after each DIIS step
- `tol::Float64`: Convergence tolerance
- `max_diis_step::Int`: Maximum number of DIIS steps
- `damping_factor::Float64`: Factor used to stabilize the DIIS procedure
"""
struct DIIS_extrapolation_alg
    M::Int
    ΔM::Int
    tol::Float64
    max_diis_step::Int
    damping_factor::Float64 
end

"""
    DIIS_extrapolation_alg(; M::Int = 10, ΔM::Int = 3, tol::Float64 = 1e-8, max_diis_step::Int = 60, damping_factor::Float64=1e-8)

Constructor for DIIS_extrapolation_alg with default parameters.
"""
function DIIS_extrapolation_alg(; M::Int = 10, ΔM::Int = 3, tol::Float64 = 1e-8, max_diis_step::Int = 60, damping_factor::Float64=1e-8) 
    return DIIS_extrapolation_alg(M, ΔM, tol, max_diis_step, damping_factor)
end

"""
    power_method_alg(; M::Int = 100, tol::Float64 = 1e-8)

Creates a DIIS configuration suitable for power method iteration without extrapolation.
"""
function power_method_alg(; M::Int = 100, tol::Float64 = 1e-8) 
    return DIIS_extrapolation_alg(M = M, ΔM = 0, tol=tol, max_diis_step = 0, damping_factor=0.0)
end

"""
    iterative_solver(_f, Xi, alg::DIIS_extrapolation_alg = DIIS_extrapolation_alg())

Solves an iterative problem using DIIS extrapolation.

# Arguments
- `_f`: Function to be iterated
- `Xi`: Initial guess
- `alg`: DIIS algorithm parameters
"""
function iterative_solver(_f, Xi, alg::DIIS_extrapolation_alg = DIIS_extrapolation_alg())
    M, ΔM, tol, max_diis_step, damping_factor = alg.M, alg.ΔM, alg.tol, alg.max_diis_step, alg.damping_factor

    subspace_errs = []
    subspace_xjs = []
    Xj = deepcopy(Xi)

    for _ in 1:M
        Xj, is_converged = iteration_step!(_f, subspace_xjs, subspace_errs, Xj, 0, tol)
        is_converged && (return Xj)
    end
    init_err = norm(subspace_errs[1])

    if max_diis_step > 0
        B = initialize_ovlpmat(subspace_errs; damping_factor=damping_factor)
        for diis_step in 1:max_diis_step
            Xj = DIIS_extrapolation(B, subspace_xjs)
            Xj = _f(Xj)
            is_converged = false
            for _ in 1:ΔM
                Xj, is_converged = iteration_step!(_f, subspace_xjs, subspace_errs, Xj, diis_step, tol; init_err=init_err)
                is_converged && (return Xj)
            end
            update_ovlpmat!(B, ΔM, subspace_errs; damping_factor=damping_factor)
        end
    end
    return Xj
end

"""
    iteration_step!(_f, subspace_xjs::AbstractVector, subspace_errs::AbstractVector, Xj, diis_step::Int, tol::Float64; init_err::Float64 = 0.0)

Performs a single iteration step in the DIIS procedure.

# Arguments
- `_f`: Function to be iterated
- `subspace_xjs`: Vector of previous solutions
- `subspace_errs`: Vector of previous errors
- `Xj`: Current solution
- `diis_step`: Current DIIS step number
- `tol`: Convergence tolerance
- `init_err`: Initial error for relative convergence check

Returns a tuple of (new solution, convergence status).
"""
function iteration_step!(_f, subspace_xjs::AbstractVector, subspace_errs::AbstractVector, Xj, diis_step::Int, tol::Float64; init_err::Float64 = 0.0)
        push!(subspace_xjs, Xj)
        (diis_step > 0) && popfirst!(subspace_xjs)

        Xj1 = _f(Xj)
        err = Xj1 - Xj 

        push!(subspace_errs, err)
        (diis_step > 0) && popfirst!(subspace_errs)

        Xj = Xj1
        norm_err = norm(err)
        printstyled("[DIIS step $(diis_step)]: err=$(norm_err) \n"; color=:light_yellow)
        is_converged = (norm_err < tol * max(1, init_err))
        if is_converged
            printstyled("geometric series converged\n"; color=:light_yellow)
        end
        return Xj, is_converged
end

"""
    real_inner(x::AbstractVector, y::AbstractVector)

Computes the real part of the inner product between two vectors.
"""
function real_inner(x::AbstractVector, y::AbstractVector)
    return real(VectorInterface.inner(x, y))
end

"""
    initialize_ovlpmat(subspace_errs::AbstractVector; damping_factor::Float64=0.0, inner=real_inner)

Initializes the overlap matrix for DIIS extrapolation.

# Arguments
- `subspace_errs`: Vector of error vectors
- `damping_factor`: Factor to stabilize the diagonal elements
- `inner`: Function to compute inner products

Returns the initialized overlap matrix with an additional row/column for the constraint.
"""
function initialize_ovlpmat(subspace_errs::AbstractVector; damping_factor::Float64=0.0, inner=real_inner)
    M = length(subspace_errs)
    B = zeros(ComplexF64, (M+1, M+1))
    for ix in 1:M
        for iy in 1:M
            B[ix, iy] = inner(subspace_errs[ix], subspace_errs[iy])
        end
    end
    B[1:M, M+1] .= 1
    B[M+1, 1:M] .= 1
    B[diagind(B)] .*= 1 + damping_factor 
    return B
end

"""
    update_ovlpmat!(B::Matrix{<:Number}, ΔM::Int, subspace_errs::AbstractVector; damping_factor::Float64=0.0, inner=real_inner)

Updates the overlap matrix in-place after new error vectors are added.

# Arguments
- `B`: Existing overlap matrix to update
- `ΔM`: Number of new vectors added
- `subspace_errs`: Vector of error vectors
- `damping_factor`: Factor to stabilize the diagonal elements
- `inner`: Function to compute inner products
"""
function update_ovlpmat!(B::Matrix{<:Number}, ΔM::Int, subspace_errs::AbstractVector; damping_factor::Float64=0.0, inner=real_inner)
    M = size(B)[1] - 1
    B[1:M-ΔM, 1:M-ΔM] .= B[ΔM+1:end-1, ΔM+1:end-1]
    for ix in M-ΔM+1:M 
        for iy in 1:ix
            B[ix, iy] = inner(subspace_errs[ix], subspace_errs[iy])
            B[iy, ix] = B[ix, iy]'
            (ix == iy) && (B[ix, iy] *= (1+damping_factor))
        end
    end
    return B
end

"""
    DIIS_extrapolation(B::Matrix{<:Number}, subspace_xjs::AbstractVector)

Performs DIIS extrapolation to generate an improved solution.

# Arguments
- `B`: Overlap matrix including constraint row/column
- `subspace_xjs`: Vector of previous solutions

Returns the extrapolated solution.
"""
function DIIS_extrapolation(B::Matrix{<:Number}, subspace_xjs::AbstractVector)
    M = size(B)[1] - 1

    v0 = zeros(ComplexF64, M+1)
    v0[M+1] = 1
    Cs = Hermitian(B) \ v0
    if ! (sum(Cs[1:M]) ≈ 1)
        @error "DIIS_extrapolation failed: sum(Cs[1:M]) = $(sum(Cs[1:M]))"
    end
    
    return sum(Cs[1:M] .* subspace_xjs[1:M])
end

