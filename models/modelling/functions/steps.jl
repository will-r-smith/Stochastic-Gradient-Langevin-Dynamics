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
function A_step(q, p, h)
    q = q + h * p
    return q
end

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
function B_step(p, q, h, grad_U, data, N, n)
    F = - grad_U(data, q, N, n)
    p = p + h * F
    return p
end

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
function O_step(p, h, phi, A, beta)
    R = randn(length(p))
    if typeof(phi) == Matrix{Float64}
        alpha = alpha = exp(- h * phi)
        p = alpha * p + sqrt.(A ./ (phi .* beta)) * sqrt.(1 .- alpha ^ 2) * R
    elseif phi == 0 
        p = p + sqrt(h) * sqrt(2 * A / beta)
    else
        alpha = alpha = exp(- h * phi)
        p = alpha * p + sqrt(A / (beta * phi)) * sqrt(1 - alpha ^ 2) * R
    end

    return p

end

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
function D_step(p, h, xi, mu, beta)
    xi = xi + (dot(p, p) - length(p) / beta) * h / mu
    return xi
end

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
function C_step(q, I, N, n, grad_U, t, data)
    g_theta_mean = grad_U(data, q, N, n)
    V_diagonal = zeros(length(q))
    
    for i in 1:length(data)
        g_theta_i = grad_U([data[i]], q, N, n) .- g_theta_mean
        V_diagonal .+= g_theta_i .^ 2
    end
    
    V = Diagonal(V_diagonal ./ (n-1))

    kappa = 1 / t
    I = (1 - kappa) * I + kappa * V

    return I
end