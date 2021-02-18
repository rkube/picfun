module pic_utils

using particles: particle
using grids: grid_1d

export S_vec, b1, smooth, deposit

@doc """
Implements the binomial smoothing operator:
SM(Q)_i = (Q_[i-1] + 2 Q[i] + Q_[i+1]) / 4

""" ->
function smooth(Q::AbstractArray{<:AbstractFloat})
  Q_sm = zeros(length(Q))
  Q_sm[2:end-1] = 0.25 * (Q[1:end-2] + 2 * Q[2:end-1] + Q[3:end])
  Q_sm[1] = 0.25 * (Q[end] + 2. * Q[1] + Q[2])
  Q_sm[end] = 0.25 * (Q[end - 1] + 2. * Q[end] + Q[1])
  return Q_sm
end

@doc """
b-spline b1.

          0  ,   if |x| > 1
b1(x) = { x+1,   if -1 < x < 0
          1-x.   if 0 <= x , 1
""" ->
function b1(z, zp, Δz)
    arg = (z - zp) / Δz
    if(abs(arg) > 1)
        return (0.0)
    end

    if (arg < 0)
        return (arg + 1)
    elseif (arg >= 0)
        return (-arg + 1)
    end
end


@doc """
Given a particle position zp, evaluate the S(x, xp) on the entire grid
""" ->
function S_vec(zp::AbstractFloat, _zgrid::AbstractArray{<:AbstractFloat}, Δz::AbstractArray{<:AbstractFloat})
    # We have periodic boundary conditions and zgrid does not include the element at L
    # The algorithm below ignores this and calculates the weights on the last index where z < zp
    # and the next index.
    # Now, when last_idx is just the last element of _zgrid, we need to have an element to the right.
    # So here we just pad.
    zgrid = deepcopy(_zgrid)
    push!(zgrid, zgrid[end] + Δz[end])
    # Find the indices of the grid points, where the current zp is less then 1 dz away
    last_idx = findall(zgrid .< zp)[end]

    S = zeros(length(zgrid))

    for i in [last_idx, last_idx + 1]
        S[i] = b1(zgrid[i], zp, Δz[last_idx])
    end

    # If the last_idx where z < zp is just the end of the array, the padded element needs to be
    # assigned to the first one
    if (last_idx == length(_zgrid))
        S[1] = S[end]
    end
    pop!(S)

    return S
end


@doc """
Evaluate the expression ∑_{particle} f(p) * b1(z_i, z_particle, Δz), where i
indices the grid, for all grid points.
""" ->
function deposit(ptl_vec::Array{particle}, zgrid::grid_1d, fun::Function)
    # Get the z-coordinates of the grid, plus one point at the upper boundary.
    zz = (0:1:zgrid.Nz) * zgrid.Δz
    # S contains the sum over all particles of f(p) * b1(z_i, z_p, Δz)
    S = zeros(zgrid.Nz)
    # Find the last grid index i where zz[i] < ptl_vec.
    # Add 1 since we have 1-based indexing
    last_idx = map(p -> 1 + Int(floor(p.pos / zgrid.Δz)), ptl_vec)

    # Don't parallelize just yet. This leads to faulty results. May have to use atomics
    # somewhere in here?
    for idx ∈ 1:length(ptl_vec)
        # gidx[01] serves two purposes:
        # 1.) Index grid quantities
        # 2.) Get the z-coordinate of the grid at that index.
        #     Do this by subtracting 1!
        gidx0 = last_idx[idx]
        # When wrapping at Nz, add one
        gidx1 = gidx0 == zgrid.Nz ? 1 : gidx0 + 1
        left_val = b1((gidx0 - 1) * zgrid.Δz, ptl_vec[idx].pos, zgrid.Δz)
        # Use gidx0 to calculate right_val instead of gidx1.
        # This captures the case, where gidx1 is 0, at the right side of the domain.
        right_val = b1(gidx0 * zgrid.Δz, ptl_vec[idx].pos, zgrid.Δz)
        #println("idx = $(idx), gidx0 = $(gidx0), gidx1 = $(gidx1), left_val=$(left_val), right_val=$(right_val)")
        if( abs(left_val + right_val - 1.0) > 1e-6)
            println("Conservation requirement broken")
            println("idx = $(idx), sum=$(left_val+right_val), left_val=$(left_val), right_val=$(right_val), gidx0=$(gidx0), gidx1=$(gidx1), x=$(ptl_vec[idx].pos)")
            break
        end
        S[gidx0] += left_val * fun(ptl_vec[idx])
        S[gidx1] += right_val * fun(ptl_vec[idx])
    end
    return(S)
end

end
