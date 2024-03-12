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
function model1_grad_U(data, q, N, n)
    # 1st model - Bayesian Inference for Gaussian Distribution
    mu, gamma = q

    sigma_x = sum(data)
    sigma_x2 = sum((data .- mu).^2)

    d_mu = (N + 1) * mu * gamma - gamma * N * sigma_x / n 
    d_gamma = 1 - (N + 1) / (2 * gamma) + mu^2 / 2 + N * sigma_x2 / (2 * n)


    return [d_mu, d_gamma]

end