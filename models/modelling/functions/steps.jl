"""
A_step(q, p, h)

Description: 
Performs one step of the A-type update

# Arguments
q::Array: Current position vector.
p::Array: Current momentum vector.
h::Float64: Step size.

# Examples
```jldoctest
julia> A_step([0.1, 0.2], [0.3, 0.4], 0.01)
2-element Array{Float64,1}:
 0.103
 0.204
```
"""
function A_step(q, p, h)
    q = q + h * p
    return q
end

"""
B_step(p, q, h, grad_U, data, N, n)

Description: 
Performs one step of the B-type update

# Arguments
p::Array: Current momentum vector.
q::Array: Current position vector.
h::Float64: Step size.
grad_U::Function: Function to compute the gradient of the potential energy.
data::Array: Data points used for the update.
N::Int: Total number of data points.
n::Int: Number of data points in the subset.

# Examples
```jldoctest
julia> B_step([0.3, 0.4], [0.1, 0.2], 0.01, grad_U, data, 100, 10)
2-element Array{Float64,1}:
 0.301...
 0.402...
```
"""
function B_step(p, q, h, grad_U, data, N, n)
    F = - grad_U(data, q, N, n)
    p = p + h * F
    return p
end

"""
O_step(p, h, phi, A, beta)

Description: 
Performs one step of the O-type update

# Arguments
p::Array: Current momentum vector.
h::Float64: Step size.
phi::Float64|Matrix{Float64}: Parameter for the update.
A::Float64: A constant.
beta::Float64: Inverse temperature.

# Examples
```jldoctest
julia> O_step([0.3, 0.4], 0.01, 0.1, 0.2, 0.5)
2-element Array{Float64,1}:
 0.3...
 0.4...
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
D_step(p, h, xi, mu, beta)

Description: 
Performs one step of the D-type update

# Arguments
p::Array: Current momentum vector.
h::Float64: Step size.
xi::Float64: Parameter for the update.
mu::Float64: A constant.
beta::Float64: Inverse temperature.

# Examples
```jldoctest
D_step([0.3, 0.4], 0.01, 0.1, 0.2, 0.5)
0.3...
```
"""
function D_step(p, h, xi, mu, beta)
    xi = xi + (dot(p, p) - length(p) / beta) * h / mu
    return xi
end

"""
C_step(q, I, N, n, grad_U, t, data)

Description: 
Performs one step of the C-type update

# Arguments
q::Array: Current position vector.
I::Matrix{Float64}: Information matrix.
N::Int: Total number of data points.
n::Int: Number of data points in the subset.
grad_U::Function: Function to compute the gradient of the potential energy.
t::Float64: Time parameter.
data::Array: Data points used for the update.

# Examples
```jldoctest
julia> C_step([0.1, 0.2], zeros(2,2), 100, 10, grad_U, 0.5, data)
2×2 Matrix{Float64}:
 0.0  0.0
 0.0  0.0
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




"""
step_function(q, p, h, integrator, steps, N, n, A, beta, data, grad_U, xi, I, t)

Description: 
Performs one step of the Langevin dynamics.

# Arguments
q::Array: Current position vector.
p::Array: Current momentum vector.
h::Float64: Step size.
integrator::String: Type of integrator.
steps::Array{String}: Array of steps to be performed.
N::Int: Total number of data points.
n::Int: Number of data points in the subset.
A::Float64: A constant.
beta::Float64: Inverse temperature.
data::Array: Data points used for the update.
grad_U::Function: Function to compute the gradient of the potential energy.
xi::Float64: Parameter for the update.
I::Matrix{Float64}: Information matrix.
t::Float64: Time parameter.

# Examples
```jldoctest
julia> step_function(q, p, h, integrator, steps, N, n, A, beta, data, grad_U, xi, I, t)
expected_output
```
"""
function step_function(q, p, h, integrator, steps, N, n, A, beta, data, grad_U, xi, I, t)
    for s in steps 
        occurences = count(x -> x == s, steps)
        if string(s) == "A"
            q = A_step(q, p, h / occurences)
        elseif string(s) == "B"
            p = B_step(p, q, h / occurences, grad_U, data, N, n)
        elseif string(s) == "O"
            if integrator == "SGLD"
                phi = A
            elseif integrator == "CCAdL"
                phi = (h / (2 * occurences)) * beta * (N^2 / n) * I .+ xi
            elseif integrator == "SGNHT"
                phi = xi
            end
            p = O_step(p, h/occurences, phi, A, beta)
        elseif string(s) == "D"
            mu = 1
            xi = D_step(p, h, xi, mu, beta)
        elseif string(s) == "C"
            I = C_step(q, I, N, n, grad_U, t, data)
        end
    end

    return q, p
end



"""
run_simulation(q0, p0, Nsteps, h, integrator, steps, subset_prop, step_function, grad_U, data, A, beta, xi0, I0)

Description: 

# Arguments
q::Array: Current position vector.
p::Array: Current momentum vector.
h::Float64: Step size.
integrator::String: Type of integrator.
steps::Array{String}: Array of steps to be performed.
N::Int: Total number of data points.
n::Int: Number of data points in the subset.
A::Float64: A constant.
beta::Float64: Inverse temperature.
data::Array: Data points used for the update.
grad_U::Function: Function to compute the gradient of the potential energy.
xi::Float64: Parameter for the update.
I::Matrix{Float64}: Information matrix.
t::Float64: Time parameter.

# Examples
```jldoctest
julia> name(args)
expected_output
```
"""
function run_simulation(q0, p0, Nsteps, h, integrator, steps, subset_prop, step_function, grad_U, data, A, beta, xi0, I0)
    q_t = zeros(length(q0),Nsteps)
    p_t = zeros(length(q0),Nsteps)
    t_t = zeros(Nsteps)

    q = copy(q0)
    p = copy(p0)
    xi = copy(xi0)
    I = copy(I0)
    t = 0.0

    N = size(data,1)
    n = Int(round(N * subset_prop))

    for i in 1:Nsteps
        idx = randperm(N)[1:n]
        samples = data[idx,:]
        q, p = step_function(q, p, h, integrator, steps, N, n, A, beta, samples, grad_U, xi, I, t)

        t += h
        
        q_t[:,i] = q
        p_t[:,i] = p
        t_t[i] = t

    end

    return q_t, p_t, t_t
end





function BAOAB_step(q, p, h, integrator, steps, N, n, A, beta, data, grad_U, xi, I, t)
    phi = A
    p = B_step(p, q, h / 2, grad_U, data, N, n)
    q = A_step(q, p, h / 2)
    p = O_step(p, h, phi, A, beta)
    q = A_step(q, p, h / 2)
    p = B_step(p, q, h / 2, grad_U, data, N, n)
    return q, p
end