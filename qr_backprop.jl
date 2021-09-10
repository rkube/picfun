



function Base.similar(a::CUDA.CUSOLVER.CuQRPackedQ{S,T}) where {S,T}
  CuArray{S, ndims(a)}(undef, size(a))
end

Zygote.@adjoint function sum(xs::CUDA.CUSOLVER.CuQRPackedQ; dims = :)
  sum(xs), Δ -> (fill!(similar(xs), Δ), )
end

Base.sum(xs::CUDA.CUSOLVER.CuQRPackedQ) = sum(CuArray(xs))


function ChainRules.rrule(::typeof(CuArray), A::CUDA.CUSOLVER.CuQRPackedQ)
    function pullback(Ȳ)
        return (NoTangent(), Ȳ)
    end
    return CuArray(A), pullback
end


function ChainRules.rrule(::Type{T}, Q::LinearAlgebra.QRCompactWYQ) where {T<:Array}
    T(Q), dy -> (NoTangent(), hcat(dy, falses(size(Q,1), size(Q,2)-size(Q.factors,2))))
end

function ChainRules.rrule(::Type{T}, Q::CUDA.CUSOLVER.CuQRPackedQ) where {T<:CuArray}
    T(Q), dy -> (NoTangent(), hcat(dy, falses(size(Q,1), size(Q,2)-size(Q.factors,2))))
end



function ChainRulesCore.rrule(::typeof(getproperty), F::CUDA.CUSOLVER.CuQR, d::Symbol)
    function getproperty_qr_pullback(Ȳ)
        ∂factors = d === :Q ? Ȳ : nothing
        ∂τ = d === :R ? Ȳ : nothing

        ∂F = Tangent{CUDA.CUSOLVER.CuQR}(; factors=∂factors, τ=∂τ)
        return (NoTangent(), ∂F)
    end

    return getproperty(F, d), getproperty_qr_pullback
end


function ChainRules.rrule(::typeof(getproperty), F::LinearAlgebra.QRCompactWY, d::Symbol)
    function getproperty_qr_pullback(Ȳ)
        ∂factors = d === :Q ? Ȳ : nothing
        ∂T = d === :R ? Ȳ : nothing

        ∂F = Tangent{LinearAlgebra.QRCompactWY}(; factors=∂factors, T=∂T)
        return (NoTangent(), ∂F)
    end

    return getproperty(F, d), getproperty_qr_pullback
end

function ChainRules.rrule(::typeof(qr), A::CuArray{T}) where {T}
    QR = qr(A)
    m, n = size(A)

    Q, R = QR
    Q_arr = CuArray(Q)
    R_arr = CuArray(R)

    function qr_pullback_cu(Ȳ::Tangent)
        Q̄ = Ȳ.factors
        R̄ = Ȳ.T

        function qr_pullback_square_deep(Q̄, R̄, Q, R)
            M = R̄*R' - Q'*Q̄
            # M <- copyltu(M)
            M = tril(M) + transpose(tril(M,-1))
            Ā = (Q̄ + Q * M) / R'
        end

        if m ≥ n
            Q̄ = Q̄ isa ChainRules.AbstractZero ? Q̄ : CuArray(Q̄[:, axes(R, 2)])
            Ā = qr_pullback_square_deep(Q̄, R̄, Q_arr, R_arr)
        else
            # partition A = [X | Y]
            # X = A[1:m, 1:m]
            Y = A[1:m, m + 1:end]

            # partition R = [U | V], and we don't need V
            U = R[1:m, 1:m]
            # V = R[1:m, m:end]
            if R̄ isa ChainRules.AbstractZero
                #@info "R̄ = 0, -> init V̄=0, Q̄_prime=0"
                V̄ = zeros(T, size(Y)) |> CuArray
                Q̄_prime = zeros(T, size(Q)) |> CuArray
                Ū = R̄
            else
                # partition R̄ = [Ū | V̄]
                Ū = R̄[1:m, 1:m]
                V̄ = R̄[1:m, m + 1:end]
                Q̄_prime = Y * V̄'
            end

            Q̄_prime = Q̄ isa ChainRules.AbstractZero ? Q̄_prime : Q̄_prime + Q̄

            @show typeof(Q̄_prime), typeof(Ū), typeof(Q_arr), typeof(U)
            X̄ = qr_pullback_square_deep(Q̄_prime, Ū, Q_arr, U)
            Ȳ = Q * V̄
            # partition Ā = [X̄ | Ȳ]
            Ā = [X̄ Ȳ]
        end

        return (NoTangent(), Ā)
    end

    return QR, qr_pullback_cu
end


function ChainRules.rrule(::typeof(qr), A::AbstractMatrix{T}) where {T}
    QR = qr(A)
    m, n = size(A)
    function qr_pullback(Ȳ::Tangent)
        function qr_pullback_square_deep(Q̄, R̄, Q, R)
            M = R*R̄' - Q̄'*Q
            # M <- copyltu(M)
            M = tril(M) + transpose(tril(M,-1))
            Ā = (Q̄ + Q * M) / R'
        end
        Q̄ = Ȳ.factors
        R̄ = Ȳ.T
        Q = QR.Q
        R = QR.R
        if m ≥ n
            # qr returns the full QR factorization, including silent columns. We need to crop them
            Q̄ = Q̄ isa ChainRules.AbstractZero ? Q̄ : Q̄[:, axes(R, 2)]
            Q = Matrix(Q)
            Ā = qr_pullback_square_deep(Q̄, R̄, Q, R)
        else    # This is the case m < n, i.e. a short and wide matrix A
            Y = @view A[1:m, m + 1:end]

            U = R[1:m, 1:m]
            if R̄ isa ChainRules.AbstractZero
                V̄ = zeros(size(Y))
                Q̄_prime = zeros(size(Q))
                Ū = R̄
            else
                # partition R̄ = [Ū | V̄]
                Ū = @view R̄[1:m, 1:m]
                V̄ = @view R̄[1:m, m + 1:end]
                Q̄_prime = Y * V̄'
            end

            Q̄_prime = Q̄ isa ChainRules.AbstractZero ? Q̄_prime : Q̄_prime + Q̄

            X̄ = qr_pullback_square_deep(Q̄_prime, Ū, A, Q, U)
            Ȳ = Q * V̄
            # partition Ā = [X̄ | Ȳ]
            Ā = [X̄ Ȳ]
        end
        return (NoTangent(), Ā)
    end
    return QR, qr_pullback
end



