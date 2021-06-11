# Encoding: UTF-8 -*-

using Flux
using DelimitedFiles
using Statistics 
using Printf
using LinearAlgebra
using MLDataPattern
using BSON: @save

push!(LOAD_PATH, pwd())

using mlutils: make_sim_onehot

# Neural network takes as input
# n(x), E(x), Δt
# and produces num_aug vectors which form
# V = [v₁, v₂, ..., vₙ]

Nz = 32
num_aug = 8

#basedir = "/global/cscratch1/sd/rkube/picfun"
basedir = "/Volumes/2TB APFS Raid/simulations/picfun/"

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

    input_fn_onehot = make_sim_onehot(f, k, A, Δt)
        
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
    x_all = cat(x_all, reshape(repeat(reshape(input_fn_onehot, (6, 1)), outer=size(E_norm, 2)), (6, size(E_norm, 2))), dims=1)

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
    @assert size(V0)[end] == size(fᵐ)[end]
    norm_sum = 0.0
    for i ∈ 1:batch_size
        V = V0[:, :, i]
        # Following (6.12) in Trefethen to get f.
        # x = pinv(transpose(V) * V) * transpose(V) * fᵐ[:, i]
        # f = V*x
        f = V * pinv(transpose(V) * V) * transpose(V) * fᵐ[:, i]
        norm_sum += norm(f - fᵐ[:, i]) / norm(fᵐ[:,i])
    end
    # Average the accumulated loss over the mini-batch
    # https://stackoverflow.com/questions/55368741/how-to-deal-with-mini-batch-loss-in-pytorch
    return(norm_sum / batch_size)
end

# Load data from two simulations with different basis functions
data_1 = load_data(basedir, sin, 1, 1e-2, 0.1);
data_2 = load_data(basedir, cos, 1, 1e-2, 0.1);
data_3 = load_data(basedir, sin, 2, 1e-2, 0.1);
data_4 = load_data(basedir, cos, 2, 1e-2, 0.1);
data_5 = load_data(basedir, sin, 1, 1e-3, 0.1);
data_6 = load_data(basedir, cos, 1, 1e-3, 0.1);
data_7 = load_data(basedir, sin, 2, 1e-3, 0.1);
data_8 = load_data(basedir, cos, 2, 1e-3, 0.1);


# Concatenate all datasets
all_x = cat(data_1[1], data_2[1], data_3[1], data_4[1], data_5[1], data_6[1], data_7[1], data_8[1], dims=2);
all_y = cat(data_1[2], data_2[2], data_3[2], data_4[2], data_5[2], data_6[2], data_7[2], data_8[2], dims=2);

all_x = getobs(shuffleobs(all_x));
all_y = getobs(shuffleobs(all_y));

# Split into train and test set
num_samples = size(all_x, 2)
idx_split = Int(num_samples * 0.8)
x_train = all_x[:, 1:idx_split];
y_train = all_y[:, 1:idx_split];

x_dev = all_x[:, idx_split+1:end];
y_dev = all_y[:, idx_split+1:end];

# Optimizer and parameters
num_epochs = 20
η = 1e-2
opt = Momentum(η, 0.99)

# Define a data loader
batch_size = 32
train_loader = Flux.Data.DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)
dev_loader = Flux.Data.DataLoader((x_dev, y_dev), batchsize=batch_size, shuffle=true)

# To get the first element, I can use 
# (x, y) = first(train_loader)
model = Chain(Dense(2 * Nz + 6, 500, swish), Dense(500, 1000, swish),  Dense(1000, 1000, swish), Dense(1000, Nz * num_aug))
params = Flux.params(model)
all_loss = zeros(num_epochs);

for e in 1:num_epochs
    for (x, y) in train_loader
        grads = Flux.gradient(params) do
            proj_loss(x, y)
        end

        Flux.Optimise.update!(opt, params, grads)
    end

    # Calculate loss on dev set
    for (x, y) in dev_loader
        all_loss[e] += proj_loss(x, y) / length(dev_loader) 
    end

    @show all_loss[e]
end

model_name = @sprintf "simple_MLP_num_pca_%02d.bson" num_aug
@save model_name model


# End of file
