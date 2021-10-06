# Encoding: UTF-8 -*-

using Flux 
using Zygote
using CUDA
using ChainRules
using ChainRulesCore
using Statistics 
using LinearAlgebra
using MLDataPattern
using DelimitedFiles
using BSON: @save
using Printf
using ArgParse
using BenchmarkTools

push!(LOAD_PATH, pwd())

using mlutils: make_sim_onehot
using parse_config: build_nn_modelname_v4

CUDA.allowscalar(false)
include("qr_backprop.jl")

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

        "--num_epochs"
            help = "Number of epochs to train"
            arg_type = Int
            default = 30
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

   end
    return parse_args(s)
end


# For the first Newton iteration, we are given n(x), E(x), Δt and try to
# match the first GMRES update, δE.
# We need to load the input, x = n(x), E(x), Δt, and the output y ≡ fᵐ = δE.
#
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
    norm_sum = 0.0f0
    for i ∈ 1:this_batch
        V = reshape(V0[:, i], (Nz, num_aug))
        Q, _ = qr(V)
        Q = CuArray(Q)
        f = Q * Q' * fᵐ[:, i]
        #f = ((Q' * fᵐ[:, i])' * Q')'
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
num_epochs = parsed_args["num_epochs"] 
prob_da = parsed_args["dropout"]
η = parsed_args["learning_rate"]
batch_size = parsed_args["batch_size"]
conv1_width = parsed_args["conv1_width"]
conv2_width = parsed_args["conv2_width"]
Nz = 32 

# Meth-programming ftw: https://discourse.julialang.org/t/convert-string-to-function-name/1547/3
activation = getfield(Main, Symbol(parsed_args["activation"]))
optim_fun = getfield(Main, Symbol(parsed_args["optimizer"])) 

#basedir = "/global/cscratch1/sd/rkube/picfun"
#basedir = "/Volumes/2TB APFS Raid/simulations/picfun/"
basedir = "/home/rkube/gpfs/picfun/DD_GMRES/baseline_Nz32"


function load_dir(basedir, f, k, A, Δt)
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

    return ne_data, E_data, δE_data, input_fn_onehot
end

sim_param_list = [(sin, 1, 1e-3, 1.0),
                  (sin, 1, 2e-3, 1.0),
                  (sin, 1, 5e-3, 1.0),
                  (sin, 1, 1e-2, 1.0),
                  (sin, 2, 1e-3, 1.0),
                  (sin, 2, 2e-3, 1.0),
                  (sin, 2, 5e-3, 1.0),
                  (sin, 2, 1e-2, 1.0),
                  (cos, 1, 1e-3, 1.0),
                  (cos, 1, 2e-3, 1.0),
                  (cos, 1, 5e-3, 1.0),
                  (cos, 1, 1e-2, 1.0),
                  (cos, 2, 1e-3, 1.0),
                  (cos, 2, 2e-3, 1.0),
                  (cos, 2, 5e-3, 1.0),
                  (cos, 2, 1e-2, 1.0)]

ne_list = []
E_list = []
δE_list = []
onehot_list = []
for sim_param ∈ sim_param_list
    ne, E, δE, onehot = load_dir(basedir, sim_param...)
    push!(ne_list, ne)
    push!(E_list, E)
    push!(δE_list, δE)
    push!(onehot_list, onehot)
end

# Create arrays where x is along the first dimension and samples are second dimension:
# all_ne[:, 1] = 1st profile
# all_E[:, 44] = 44th profile 
all_ne = hcat(ne_list...)
all_E = hcat(E_list...)
all_δE = hcat(δE_list...)

# Scale to the magic numbers
all_ne = (all_ne .- 1.0) ./ 0.0045
all_E = (all_E .- 0.0) ./ 0.0028
all_δE = (all_δE .- 0.0) ./ 0.00246


# Blow-up the onehot vectors so that they  correspond in sample-size to the individual simulations
# Let 
# nsample = size(all_ne[i], 2) 
# the number of profiles sampled in the i-th simulation.
# Then we repeat the one-hot vector for that simulatione exactly nsample times.
all_onehot = [repeat(reshape(x, (6,1)), 1, size(ne, 2)) for (x, ne) in zip(onehot_list, ne_list)]
all_onehot = hcat(all_onehot...)

# Now we can vertically stack ne, E, and onehot so that each row (1st) dimension is all one sample.
all_x = Float32.(vcat(all_ne, all_E, all_onehot)) |> gpu;
all_y = Float32.(all_δE) |> gpu;

# Concatenate all datasets
all_x = getobs(shuffleobs(all_x));
all_y = getobs(shuffleobs(all_y));

# Split into train and test set
num_samples = size(all_x, 2)
idx_split = Int((num_samples * 0.8) ÷ 1)
x_train = all_x[:, 1:idx_split];
y_train = all_y[:, 1:idx_split];

x_dev = all_x[:, idx_split+1:end];
y_dev = all_y[:, idx_split+1:end];

# Optimizer and parameters
opt = optim_fun(η)

# Define a data loader
train_loader = Flux.Data.DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)
dev_loader = Flux.Data.DataLoader((x_dev, y_dev), batchsize=batch_size, shuffle=true)

# To get the first element, I can use 
# (x, y) = first(train_loader)
# Now try a 1d-cnn

channels_1 = 4
channels_2 = 16
num_dense = 32

model_cnn = Chain(Conv((conv1_width, 1), 2=>channels_1, activation), 
                  Conv((conv2_width, 1), channels_1=>channels_2, activation), 
                  Flux.flatten)
model_par = Chain(Dense(6, num_dense, activation), 
                  Dropout(prob_da),
                  Dense(num_dense, num_dense, activation))

num_intermediate = channels_2 * (Nz - sum([2*(x÷2) for x in [conv1_width conv2_width ]]))

model = Chain(Parallel(vcat, model_cnn, model_par), 
              Dropout(prob_da),
              Dense(num_intermediate + num_dense, Nz * num_aug, activation),
              Dropout(prob_da),
              Dense(Nz * num_aug, Nz * num_aug, activation)) |> gpu

params = Flux.params(model);
all_loss = zeros(num_epochs);

@time for e in 1:num_epochs
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

    @show e, all_loss[e]
end

@show model

model = model |> cpu


cfg = Dict("num_aug" => num_aug, "optim_fun" => optim_fun, "batch_size" => batch_size, "activation" => optim_fun, "conv1_width" => conv1_width, "conv2_width" => conv2_width)
model_name = build_nn_modelname_v4(cfg)


#model_name = @sprintf "nnv4_naug%02d_%s_bs%03d_%s_c%02d%02d.bson" num_aug optim_fun batch_size activation conv1_width conv1_width
@save model_name model


# End of file
