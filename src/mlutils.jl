#module mlutils

export make_sim_onehot

function make_sim_onehot(f, k, A,  Δt)
    # Build 1-hot vector for input function
    input_fn_1hot = zeros(6)
    if f == sin
        if k == 1
            input_fn_1hot[1] = 1.0
        elseif k == 2
            input_fn_1hot[3] = 1.0
        end
    elseif f == cos
        if k == 1
            input_fn_1hot[2] = 1.0
        elseif k == 2
            input_fn_1hot[4] = 1.0
        end
    end
    input_fn_1hot[5] = A
    input_fn_1hot[6] = Δt
    return(input_fn_1hot)
end


#end