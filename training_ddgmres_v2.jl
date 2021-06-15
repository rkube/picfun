# Encoding: UTF-8 -*-

using Flux, ParameterSchedulers
using ParameterSchedulers: Scheduler
using DelimitedFiles
using Statistics 
using Printf
using LinearAlgebra
using MLDataPattern
using BSON: @save
using ArgParse

push!(LOAD_PATH, pwd())

using mlutils: make_sim_onehot

# Neural network takes as input
# n(x), E(x), Δt
# and produces num_aug vectors which form
# V = [v₁, v₂, ..., vₙ]

function parse_commandline()

s = ArgParseSettings()
    @add_arg_table s begin 
        "--num_aug"
            help = "Number of augmentation vectors to learn (int)"
            arg_type = Int
            default = 8
            required = true

        "--optimizer"
            help = "Optimizer to use: ADAM, SGD, RMSProp, NADAM, OADAM, ADAGrad"
            arg_type = String
            default = "ADAM"
            required = true

        "--learning_rate"
            help = "Initial learning rate"
            arg_type = Float64
            default = 0.001
            required = true

        "--batch_size"
            help = "Batch size"
            arg_type = Int
            default = 16
            required = true

        "--dropout"
            help = "Dropout rate"
            arg_type = Float64
            default = 0.15
            required = true

        "--activation"
            help = "Activation function"
            arg_type = String
            default = "relu"
            required = true

        "--conv1_width"
            help = "Width for 1st convolution"
            arg_type = Int
            default = 3
            required = true

        "--conv2_width"
            help = "Width for 2nd convolution"
            arg_type = Int
            default = 3
            required = true

        "--conv3_width"
            help = "Width for 3rd convolution"
            arg_type = Int
            default = 5
            required = true

        "--conv4_width"
            help = "Width for 4th convolution"
            arg_type = Int
            default = 7
            required = true
    end
    return parse_args(s)
end


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
    # while batch_size is defined above, the loader may have fewer
    # samples in the last batch. Use size(model(x))[end] for the last dimension.
    this_batch = size(x, 2)
    V0 = model((reshape(x[1:2 * Nz, 1:this_batch], (Nz, 1, 2, this_batch)), x[2 * Nz + 1:end, :]))
    # V0 = reshape(model(x), (Nz, num_aug, size(model(x))[end]))
    @assert size(V0)[end] == size(fᵐ)[end]
    # Make explicitly Float32
    norm_sum = Float32.(0.0)
    for i ∈ 1:this_batch
        V = reshape(V0[:, i], (Nz, num_aug))
        # Following (6.12) in Trefethen to get f.
        # x = pinv(transpose(V) * V) * transpose(V) * fᵐ[:, i]
        # f = V*x
        f = V * pinv(transpose(V) * V) * transpose(V) * fᵐ[:, i]
        # Calculate the inverse of tr(V)*V via the SVD: https://www.johndcook.com/blog/2018/05/05/svd/
        # Since pinv does not work on gpu: https://github.com/JuliaGPU/CUDA.jl/issues/883
        # F = svd(transpose(V) * V)
        # inv_Vtr_V =  F.V * Diagonal(1.0f0 ./ F.S) * transpose(F.U)
        # f = V * inv_Vtr_V * transpose(V) * fᵐ[:, i]
        norm_sum += norm(f - fᵐ[:, i]) / norm(fᵐ[:,i])
    end
    # Average the accumulated loss over the mini-batch
    # https://stackoverflow.com/questions/55368741/how-to-deal-with-mini-batch-loss-in-pytorch
    return(norm_sum / batch_size)
end

parsed_args = parse_commandline()
println("Parsed args:")
for (arg,val) in parsed_args
    println("  $arg  =>  $val")
end

# Extract parameters from parsed_args

num_aug = parsed_args["num_aug"]       
 
prob_da = parsed_args["dropout"]
η = parsed_args["learning_rate"]
batch_size = parsed_args["batch_size"]
conv1_width = parsed_args["conv1_width"]
conv2_width = parsed_args["conv2_width"]
conv3_width = parsed_args["conv3_width"] 
conv4_width = parsed_args["conv4_width"]
Nz = 32

# Meth-programming ftw: https://discourse.julialang.org/t/convert-string-to-function-name/1547/3
activation = f = getfield(Main, Symbol(parsed_args["activation"]))
optim_fun = getfield(Main, Symbol(parsed_args["optimizer"])) 

#basedir = "/global/cscratch1/sd/rkube/picfun"
#basedir = "/Volumes/2TB APFS Raid/simulations/picfun/"
basedir = "/home/rkube/gpfs/picfun/DD_GMRES/baseline"

# Load data from two simulations with different basis functions
data_1 = load_data(basedir, sin, 1, 1e-2, 1.0);
data_2 = load_data(basedir, cos, 1, 1e-2, 1.0);
data_3 = load_data(basedir, sin, 2, 1e-2, 1.0);
data_4 = load_data(basedir, cos, 2, 1e-2, 1.0);
data_5 = load_data(basedir, sin, 1, 1e-3, 1.0);
data_6 = load_data(basedir, cos, 1, 1e-3, 1.0);
data_7 = load_data(basedir, sin, 2, 1e-3, 1.0);
data_8 = load_data(basedir, cos, 2, 1e-3, 1.0);

# Concatenate all datasets
all_x = Float32.(cat(data_1[1], data_2[1], dims=2));#, data_3[1], data_4[1], data_5[1], data_6[1], data_7[1], data_8[1], dims=2));
all_y = Float32.(cat(data_1[2], data_2[2], dims=2));#, data_3[2], data_4[2], data_5[2], data_6[2], data_7[2], data_8[2], dims=2));

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
num_epochs = 1
opt = optim_fun(η)

# Define a data loader
train_loader = Flux.Data.DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)
dev_loader = Flux.Data.DataLoader((x_dev, y_dev), batchsize=batch_size, shuffle=true)

# To get the first element, I can use 
# (x, y) = first(train_loader)
# Now try a 1d-cnn

model_cnn = Chain(Conv((conv1_width, 1), 2=>8, activation),  
                  Conv((conv2_width, 1), 8=>32, activation), Dropout(prob_da),
                  Conv((conv3_width, 1), 32=>128, activation), Dropout(prob_da), 
                  Conv((conv4_width, 1), 128=>256, activation), Flux.flatten)
model_par = Chain(Dense(6, 32, activation), Dense(32, 32, activation))

num_dense = 256 * (32 - sum([2*(x÷2) for x in [conv1_width conv2_width conv3_width conv4_width]]))

model = Chain(Parallel(vcat, model_cnn, model_par), 
              Dense(num_dense + 32, 2048, activation), 
              Dense(2048, 1024, activation), 
              Dense(1024, Nz * num_aug)) 

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

    @show e all_loss[e]
end

#model_name = @sprintf "simple_MLP_num_pca_%02d_dt1.bson" num_aug
#@save model_name model


# End of file
