# Encodin: UTF-8 -*-

module solvers

using Zygote
using FFTW: fft, ifft
using LinearAlgebra
using grids: grid_1d
using IterativeSolvers
using Printf
using DelimitedFiles

export ∇⁻², invert_laplace, dd_gmres, dd_gmres_ben


@doc """Given a charge distribution on a grid, solves for the electrostatic
potential.
""" ->
function ∇⁻²(y::Array{Float64}, zgrid::grid_1d)
	# Calculate wave number vector
	Nz, Lz = zgrid.Nz, zgrid.Lz

	k = zeros(Nz)
	k[2:Nz ÷ 2  + 1] = collect(1:Nz ÷ 2) .* 2π / Lz
	k[Nz ÷ 2 + 2:Nz] = -collect(Nz ÷ 2 - 1:-1:1) .* 2π / Lz
	# Zero out the zero-mode 
	y_ft = -fft(y) ./ k ./ k
	y_ft[1] = 0.0
	d2y = real(ifft(y_ft))
	return d2y
end

@doc """Inverts the laplace equation on a 1d grid.

""" -> 
function invert_laplace(y,  zgrid::grid_1d)
    # Solve Laplace equation
    # ρ: RHS function
    # Nz: Number of grid points
	# Lz: Length of the domain
	Nz, Lz = zgrid.Nz, zgrid.Lz
    Δz = Lz / Nz
    #zrg = range(0.0, step=Δz, length=Nz)
    #zrg = 0.0:Δz:Lz-Δz
    # Squeeze a minus in here.
    invΔz² = 1.0 / Δz / Δz

    # There should be an Eigenvalue around zero, i.e. the matrix is not invertible.
    # Periodic boundary conditions imply that the mean of the solution to is not fixed.
    # solutions can differ by a constant.
    # We can get around this by fixing phi(x0) = c:
    # http://www-m16.ma.tum.de/foswiki/pub/M16/Allgemeines/StefanPossanner/Poisson1D_FD.html#19
    # So the first row is just [1, 0, 0, ...0]

    A0 = Zygote.Buffer(zeros(Nz, Nz))
    for n ∈ 1:Nz
        for m ∈ 1:Nz
            A0[n,m] = 0.0
        end
    end
    for n ∈ 2:Nz
        A0[n, n] = -2.0 * invΔz²
        A0[n, n-1] = invΔz²
        A0[n-1, n] = invΔz²
    end
    A0[1,1] = 1.0
    A0[1, 2] = 0.0
    A0[Nz,1] = invΔz²

    A = copy(A0)
    yvec = vcat([0.0], y[2:Nz])

    ϕ_num = A \ yvec

    return(ϕ_num)
end

# Naive GMRES implementation


# Iteratively generates an orthonormal basis of a Krylov subspace
# See p. 252 from Trefethen
# and https://en.wikipedia.org/wiki/Arnoldi_iteration
function arnoldi_iteration(A, b, n=10)
    # A: m*m Array
    # b: Initial vector (m)
    # n: Dimension of Krylov subspace

    # Make sure the dimensions match
    @assert size(A)[1] == size(A)[2] == size(b)[1]
    # Dimension of the system
    m = size(A)[1]

    # n+1 orthonormal row-vectors of size m of the Krylov subspace
    Q = zeros((m, n + 1))
    # Coefficients hᵢⱼ of the (n+1, n) Hessenberg Matrix
    H = zeros((n + 1, n))

    Q[:, 1] = b / norm(b)
    # Create a view
    # https://discourse.julialang.org/t/could-you-explain-what-are-views/17535
    q = Q[:, 1]
    for j ∈ 1:n  # Iterate to n-1 and write to column j+1
        v = A * q # Generate a new candidate vector
        for i ∈ 1:j # Subtract projection on previous vectors
            H[i, j] =  conj(Q[:, i])' * v
            v = v - H[i, j] * Q[:, i]
        end
        H[j+1, j] = norm(v)
        # We need to be a bit generous with ϵ here.
        if abs(H[j+1, j]) ≤ 10 * eps(eltype(H))
            break 
        end
        q = v / H[j+1, j] # Append the produced vector to the Q-matrix
        Q[:, j+1] = q
    end

    # Return matrices Qᵢ₊₁, Hᵢ,ᵢ₊₁
    return Q, H
end

@doc """
    gmres_augment

Implements augmented GMRES: Trivedi et al. https://www.nature.com/articles/s41598-019-56212-5

A: DxD Matrix
b: D vector
V: DxN Matrix
i: Size of Krylov-space  Kᵢ(Ã, b̃). Needs to be smaller than D

Returns:
x: Least-squares solution of Ax=b over range(V) ⊕ Kᵢ(Ã, b̃)

""" ->
function dd_gmres(A, b, V, i)
    N = size(V)[2]
    # Calculate the QR decomposition of AV
    C, R = qr(A*V)
    # Julia gives the full QR decomposition. We are interested only in the
    # reduced QR factorization, considering only the first N columns
    C = C[:, 1:N]
    Cstar = adjoint(C)

    # Calculate the orthogonal projection out of span(A*v₁, A*v₂, ...)
    Pperp = one(A) - C*Cstar

    # Verify equation (4): Calculate Arnoldi decomposition using i vectors.
    Ã = Pperp * A
    b̃ = Pperp * b
    Q, H = arnoldi_iteration(Ã, b̃, i)
    # At this point, the columns in Ã should be perpendicular to the columns in A*V:
    # A*V \ Ã[:,j] ≈ 0 for every column j

    # With Q and H at hand we can move on to solving a least-squares problem
    # Solve the unconstrained least-squares problem Eq. (9) in the paper:
    # * Build the matrix C* A Q
    MM = Cstar * A * Q[:, 1:i]
    # Construct the matrices A and b 
    Alsq = [Matrix(I, N, N) MM; zeros((i+1, N)) H]
    blsq = [Cstar * b; adjoint(Q) * b]

    # Solve the least-squares problem using by calculating the reduced QR factorization
    Q2, R2 = qr(Alsq)
    Q2 = Q2[:, 1:N+i]
    # Calculate the least-squares solution
    sollsq = R2 \ (adjoint(Q2[:, 1:N+i]) * blsq)
    # Calculate the desired fᵢ from the least-squares solution
    xlsq = V * pinv(R) * sollsq[1:N] + Q[:, 1:i] * sollsq[N+1:end]
    return xlsq
end


function dd_gmres_ben(A, b, V, num_it; log=false)
    # Calculate MVPs of a and all column vectors in V
    N = size(V, 2)
    t = @elapsed begin
        AVs = [A*V[:, col] for col ∈ 1:N]
    end
    println("        DD-GMRES: A*V took $(t)s")
    # Build the matrix 
    # <Av1, Av1> <Av1, Av2>, ..., <Av1, Avn>;
    # <Av2, Av1> <Av1, Av2>, ..., <Av2, Avn>
    # ...
    # <Avn, Av1> <Avn, Av2>, ..., <Avn, Avn>
    # and the vector
    # [<Av1, b>; <Av2, b>; ...; <Avn, b>]
    AVmat = zeros(N, N)
    AVbvec = zeros(N)
    t = @elapsed begin
        for i ∈ 1:N
            # Calculate only the upper triangle
            for j ∈ 1:i 
                AVmat[i, j] = AVs[i]' * AVs[j]
                # And fill in the lower triangle
                if i != j
                    AVmat[j, i] = AVmat[i, j]
                end
            end
            # Compute entry for <Avi,b>
            AVbvec[i] = AVs[i]' * b
        end
        αvec = AVmat \ AVbvec
        # Calculate btilde
        b̃ = b - sum([αvec[i] * AVs[i] for i ∈ 1:N])
    end # @elapsed
    println("       DD-GMRES: building b̃ took $(t)s")

    t = @elapsed begin
        x̃ = gmres(A, b̃, log=log, verbose=true)
    end
    println("       DD-GMRES: solving x̃ tool $t(s)")
    if log
        x̃, gmres_log = x̃
                    # Store convergence history
        fname = @sprintf "DDGMRES_iter_%04d_convhist.txt" num_it
        open(fname, "a") do io
            writedlm(io, [norm(b̃);  gmres_log.data[:resnorm]]')
        end
    end
    x = x̃ + sum([αvec[i] * V[:,i] for i ∈ 1:N])

    # Return either a tuple (x, gmres_log), or just x
    log ? (x, gmres_log) : x
end


end # module
# End of file solvers.jl
