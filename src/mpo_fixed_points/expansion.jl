# implemetation of the error estimatation and the expansion method in [Zauner-Stauber et al. Phys. Rev. B 97 (2018) Appendix A, B]

function two_site_variation(AL::MPSTensor, AR::MPSTensor, C::MPSBondTensor, EL::EnvTensorL, ER::EnvTensorR, T::MPOTensor)
    NL = leftnull(AL)
    NR = rightnull!(permute(AR, ((1, ), (3, 2))))

    @tensoropt (a => D^4, b => D^4, c => D^2, d => D^2, e => D^2, f => D^2, g => D^2, h => D^2, i => D^2, j => D^2, k => D^2, l => D^2, m => D^2, n => D^2, o => D^2) begin
        B2[a; b] := conj(NL[c d; a]) * conj(NR[b; f e]) * EL[c; j g] * T[g d; n h] * T[h e; o i] * ER[m i; f] * AL[j n; k] * C[k; l] * AR[l o; m]
    end
    return B2
end

function changebonds!(AL::MPSTensor, AR::MPSTensor, C::MPSBondTensor, B2::MPSBondTensor, trscheme::TruncationScheme)

    U, _, V = tsvd!(B2; trunc = trscheme)
    NL = leftnull(AL)
    NR = rightnull!(permute(AR, ((1, ), (3, 2))))
    ΔAL = NL * U 
    ΔAR = V * NR
    @show space(V, 1), space(U, 2)

    # copied from MPSKit.jl: https://github.com/QuantumKitHub/MPSKit.jl/blob/d30ef9e97dec9375b43574c9877820e8922574f0/src/algorithms/changebonds/changebonds.jl#L23-L43
    # modified the code to fit the convention of the current project
    
    # update AL: add vectors, make room for new vectors:
    # AL -> [AL expansion; 0 0]
    al = _transpose_tail(catdomain(AL, ΔAL))
    lz = zerovector!(similar(al, _lastspace(ΔAL)' ← domain(al)))
    AL_updated = _transpose_front(catcodomain(al, lz))

    # update AR: add vectors, make room for new vectors:
    # AR -> [AR 0; expansion 0]
    ar = _transpose_front(catcodomain(_transpose_tail(AR), ΔAR))
    rz = zerovector!(similar(ar, codomain(ar) ← _firstspace(ΔAR)))
    AR_updated = catdomain(ar, rz)

    # update C: add vectors, make room for new vectors:
    # C -> [C 0; 0 expansion]
    l = zerovector!(similar(C, codomain(C) ← _firstspace(ΔAR)))
    C_updated = catdomain(C, l)
    r = zerovector!(similar(C, _lastspace(ΔAL)' ← domain(C_updated)))
    C_updated = catcodomain(C_updated, r)

    # update AC: recalculate
    AC_updated = AL_updated * C_updated

    return AL_updated, AR_updated, C_updated, AC_updated
end