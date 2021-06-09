# Encoding: UTF-8 -*-

using Flux
using DelimitedFiles
using Statistics 
using Printf
using LinearAlgebra

# Neural network takes as input
# n(x), E(x), Δt
# and produces num_aug vectors which form
# V = [v₁, v₂, ..., vₙ]

Nz = 32
num_aug = 8
model = Chain(Dense(2 * Nz + 6, 150, relu), Dense(150, 500, relu), Dense(500, 500, relu), Dense(500, 150, relu), Dense(150, Nz * num_aug))

basedir = "/Users/ralph/source/repos/picfun/simulations/"

f = "sin"
k = 1
A = 1e-2
Δt = 0.1


# For the first Newton iteration, we are given n(x), E(x), Δt and try to
# match the first GMRES update, δE.
# We need to load the input, x = n(x), E(x), Δt, and the output y ≡ fᵐ = δE.
#
function load_data(basedir, f, k, A, Δt)
    # basedir: Basis directory
    # f: perturbation function, f="sin","cos"
    # k: wavelength of initial perturbation, k=1, 2
    # A: initial amplitude, Float: A="1e-2", "1e-3"
    # Δt: Timestep of simulation, Float either "0.1" or "1"

    A_str = @sprintf "%1.0e" A
    A_str = replace.(A_str, s"0" => "")
    Δt_str = ""
    if Δt == 1e-1
        Δt_str *= "01"
    elseif Δt == 1.0
        Δt_str *= "1"
    end

    simdir = @sprintf "pert_%s%1d_%s_dt%s" f k A_str Δt_str 
    @show simdir

    # Build 1-hot vector for input function
    input_fn_1hot = zeros(6)
    if f == "sin"
        if k == 1
            input_fn_1hot[1] = 1.0
        elseif k == 2
            input_fn_1hot[3] = 1.0
        end
    elseif f == "cos"
        if k == 1
            input_fn_1hot[2] = 1.0
        elseif k == 2
            input_fn_1hot[4] = 1.0
        end
    end
    input_fn_1hot[5] = A
    input_fn_1hot[6] = Δt
    @show input_fn_1hot
        
    # Transpose the arrays. That way, the last dimension is the sample dimension.
    # This layout is common for flux: https://fluxml.ai/Flux.jl/stable/data/dataloader/#DataLoader
    # Also, skip the first row in E_data and ne_data as this is written before the time-integration loop
    E_data = transpose(readdlm(basedir * "/" * simdir * "/E.txt")[2:end, 2:end])
    ne_data = transpose(readdlm(basedir * "/" * simdir * "/ne.txt")[2:end, 2:end])
    δE_data = transpose(readdlm(basedir * "/" * simdir * "/GMRES_iter_0001_deltaE.txt")[1:end, 2:end])

    E_norm = (E_data .- minimum(E_data)) / (maximum(E_data) - minimum(E_data))
    ne_norm = (ne_data .- minimum(ne_data)) / (maximum(ne_data) - minimum(ne_data))
    δE_norm = (δE_data .- minimum(δE_data)) / (maximum(δE_data) - minimum(δE_data))

    # Feature vector is ne and E
    x_all = cat(ne_norm, E_norm, dims=1)
    # Now we add the simulation parameters to x_all, in the first dimension
    x_all = cat(x_all, reshape(repeat(reshape(input_fn_1hot, (6, 1)), outer=size(E_norm, 2)), (6, size(E_norm, 2))), dims=1)

    # Target vector is δE
    y_all = δE_norm

    return (x_all, y_all)

    # @show size(E_data)
end





# Define projection loss function.
# It takes input the proposed vectors v₁,...,vₙ and calculates
# min ||f - f⁽ᵐ⁾||² / ||f⁽ᵐ⁾||²
function proj_loss(x, fᵐ)

    V0 = reshape(model(x), (Nz, num_aug, batch_size))
    norm_sum = 0.0
    for i ∈ 1:size(V0, 3)
        V = V0[:, :, i]
        # Following (6.12) in Trefethen to get f.
        x = pinv(transpose(V) * V) * transpose(V) * fᵐ
        f = V*x
        norm_sum += norm(f - fᵐ) / norm(fᵐ)
    end
    return(norm_sum)
end

# Load data from two simulations with different basis functions
data_1 = load_data(basedir, "sin", 1, 1e-2, 0.1);
data_2 = load_data(basedir, "cos", 1, 1e-2, 0.1);
data_3 = load_data(basedir, "sin", 2, 1e-2, 0.1);
data_4 = load_data(basedir, "cos", 2, 1e-2, 0.1);


all_x = cat(data_1[1], data_2[1], data_3[1], data_4[1], dims=2);
all_y = cat(data_1[2], data_2[2], data_3[2], data_4[2], dims=2);



# Optimizer and parameters
num_epochs = 20
η = 1e-3
opt = ADAM(η)

# Define a data loader
batch_size = 1
train_loader = Flux.Data.DataLoader((all_x, all_y), batchsize=batch_size, shuffle=true)

# To get the first element, I can use 
# (x, y) = first(train_loader)

all_loss = zeros(num_epochs)

for e in 1:num_epochs
    current_loss = 0.0
    for (x, y) in train_loader
        grads = Flux.gradient(Flux.params(model)) do
            proj_loss(x, y)
        end
        current_loss += proj_loss(x, y)

        Flux.Optimise.update!(opt, Flux.params(model), grads)

    end
    all_loss[e] = current_loss / length(train_loader)
    @show all_loss[e]
end




# End of file