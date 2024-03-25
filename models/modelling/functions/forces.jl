"""
    name(args)

Description: 

# Arguments
- 'arg::Type': Desc
- 'arg::Type': Desc

# Examples
```jldoctest
julia> name(args)
expected_output
```
"""
function problem1_grad_U(data, q, N, n)
    # 1st model - Bayesian Inference for Gaussian Distribution
    mu, gamma = q

    sigma_x = sum(data)
    sigma_x2 = sum((data .- mu).^2)

    d_mu = (N + 1) * mu * gamma - gamma * N * sigma_x / n 
    d_gamma = 1 - (N + 1) / (2 * gamma) + mu^2 / 2 + N * sigma_x2 / (2 * n)


    return [d_mu, d_gamma]

end



function problem2_grad_U(data, w, N, n)
    # 2nd model - Large Scale Bayesian Logistic Regression
    x = data[:, 1:size(data, 2)-1]
    y = data[:, size(data, 2)]

    w_old = w

    sum = zeros(size(data, 2)-1)
    for i in 1:length(y)
        a = exp(- y[i] * dot(w, x[i, :]))
        #sum = sum .+ y[i] * x[i, :] * a / (1 + a)
        sum = sum .+ (y[i] * x[i, :] / (1 + a))
    end
    #w = w .- N/n * sum

    w = sign.(w) .* sum * N/n

    #for i in 1:length(y)
        #a = exp(- y[i] * dot(w_old, x[i, :]))
        #w = w .- (N / n * y[i] * x[i, :] * a / (1 + a))
    #end

    return w

end