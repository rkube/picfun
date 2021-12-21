# Function that build things from the configuration file
module parse_config
    using Printf

    export build_init_func

    function build_init_func(config::Dict)
            f1 = getfield(Main, Symbol(config["f1"]))
            f2 = getfield(Main, Symbol(config["f2"]))
            init_func(x) = config["A"] * (config["a1"] * f1(config["k1"] * x) +
                                          config["a2"] * f2(config["k2"] * x))
        return init_func
    end

    function build_nn_modelname_v4(config::Dict)
            model_fname = @sprintf "nnv4_naug%02d_%s_bs%03d_%s_c%02d%02d.bson" config["num_aug"] config["optim_fun"] config["batch_size"] config["activation"] config["conv1_width"] config["conv1_width"]
            model_fname = config["model_dir"] * "/" * model_fname
    end
        
    function build_nn_modelname_v6(config::Dict)
            model_fname = @sprintf "nnv6_naug%02d_%s_bs%03d_%s_c%02d%02d_lbd%7.5f_dropout%3.1f.bson" config["num_aug"] config["optim_fun"] config["batch_size"] config["activation"] config["conv1_width"] config["conv1_width"] config["lambda"] config["dropout"]
            model_fname = config["model_dir"] * "/" * model_fname
    end
        

end
