# Encoding: UTF-8 -*-

using Flux 
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

CUDA.allowscalar(false)

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



function ChainRules.rrule(::typeof(getproperty), F::LinearAlgebra.QRCompactWY, d::Symbol)
    function getproperty_qr_pullback(Ȳ)
        # The QR factorization is calculated from `factors` and T, matrices stored in the QRCompactWYQ format, see
        # R. Schreiber and C. van Loan, Sci. Stat. Comput. 10, 53-57 (1989).
        # Instead of backpropagating Q̄ and R̄ through (factors)bar and T̄, we re-use factors to carry Q̄ and T to carry R̄
        # in the Tangent object.
        ∂factors = if d === :Q
            # Here we need to bring the incoming gradient, Ȳ, into the shape of Q.
            # Now this is tricky because Q calculated from the QRCompactWYQ is not the same as matrix(Q),
            # see discussion here https://github.com/JuliaLang/julia/blob/6aaedecc447e3d8226d5027fb13d0c3cbfbfea2a/stdlib/LinearAlgebra/src/qr.jl#L335
            # As a first step, let's make the incoming gradient the same size as qr.factors
            Ȳ   
        else
            nothing
        end 

        ∂T = d === :R ? Ȳ : nothing

        ∂F = Tangent{LinearAlgebra.QRCompactWY}(; factors=∂factors, T=∂T)
        return (NoTangent(), ∂F)
    end

    return getproperty(F, d), getproperty_qr_pullback
end



function ChainRules.rrule(::typeof(qr), A::AbstractMatrix{T}) where {T}
    QR = qr(A)
    m, n = size(A)
    function qr_pullback(Ȳ::Tangent)
        # For square (m=n) or tall and skinny (m >= n), use the rule derived by 
        # Seeger et al. (2019) https://arxiv.org/pdf/1710.08717.pdf
        #   
        # Ā = [Q̄ + Q copyltu(M)] R⁻ᵀ
        #   
        # where copyltU(C) is the symmetric matrix generated from C by taking the lower triangle of the input and
        # copying it to its upper triangle : copyltu(C)ᵢⱼ = C_{max(i,j), min(i,j)}
        #   
        # This code is re-used in the wide case and therefore in a separate function.

        function qr_pullback_square_deep(Q̄, R̄, A, Q, R)
            M = R*R̄' - Q̄'*Q
            # M <- copyltu(M)
            M = tril(M) + transpose(tril(M,-1))
            Ā = (Q̄ + Q * M) / R'
        end

        # For the wide (m < n) case, we implement the rule derived by
        # Liao et al. (2019) https://arxiv.org/pdf/1903.09650.pdf
        #   
        # Ā = ([Q̄ + V̄Yᵀ] + Q copyltu(M)]U⁻ᵀ, Q V̄)
        # where A=(X,Y) is the column-wise concatenation of the matrices X (n*n) and Y(n, m-n).
        #  R = (U,V). Both X and U are full rank square matrices.
        #   
        # See also the discussion in https://github.com/JuliaDiff/ChainRules.jl/pull/306
        # And https://github.com/pytorch/pytorch/blob/b162d95e461a5ea22f6840bf492a5dbb2ebbd151/torch/csrc/autograd/FunctionsManual.cpp 
        Q̄ = Ȳ.factors
        R̄ = Ȳ.T
        Q = QR.Q
        R = QR.R
        if m ≥ n
            # qr returns the full QR factorization, including silent columns. We need to crop them 
            Q̄ = Q̄ isa ChainRules.AbstractZero ? Q̄ : Q̄[:, axes(R, 2)]
            Q = Matrix(Q)
            Ā = qr_pullback_square_deep(Q̄, R̄, A, Q, R)
        else    # This is the case m < n, i.e. a short and wide matrix A
            @warn "The qr-pullback for matrices where m<n is not covered by unit tests"
            # partition A = [X | Y]
            # X = A[1:m, 1:m]
            Y = @view A[1:m, m + 1:end]

            # partition R = [U | V], and we don't need V
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




# Define projection loss function.
# It takes input the proposed vectors v₁,...,vₙ and calculates
# min ||f - f⁽ᵐ⁾||² / ||f⁽ᵐ⁾||² with f ∈ span(V)
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
        Q, _ = qr(V)
        f = Q[:, 1:num_aug] * Q[:, 1:num_aug]' * fᵐ[:, i]

        norm_sum += mean( (f .- fᵐ[:, i]).^2) / mean( (fᵐ[:,i]).^2)
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
activation = getfield(Main, Symbol(parsed_args["activation"]))
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
all_x = Float32.(cat(data_1[1], data_2[1], data_3[1], data_4[1], data_5[1], data_6[1], data_7[1], data_8[1], dims=2)) ;
all_y = Float32.(cat(data_1[2], data_2[2], data_3[2], data_4[2], data_5[2], data_6[2], data_7[2], data_8[2], dims=2)) ;

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
num_epochs = 30
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
              Dense(num_dense + 32, Nz * num_aug, activation))
#              Dense(num_dense + 32, 2048, activation), 
#              Dense(2048, 1024, activation), 
#              Dense(1024, Nz * num_aug)) 

params = Flux.params(model)
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

#model_name = @sprintf "simple_MLP_num_pca_%02d_dt1.bson" num_aug
#@save model_name model


# End of file
