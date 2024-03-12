"""
    step(q, p, h, A, beta, force[, xi])

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
function step_function(q, p, h, integrator, steps, N, n, A, beta, data, grad_U, xi, I, t)
    for s in steps 
        occurences = count(x -> x == s, steps)
        if s == "A"
            q = A_step(q, p, h / occurences)
        elseif s == "B"
            p = B_step(p, q, h / occurences, grad_U, data, N, n)
        elseif s == "O"
            if integrator == "SGLD"
                phi = A
            elseif integrator == "CCAdL"
                phi = (h / (2 * occurences)) * beta * (N^2 / n) * I .+ xi
            elseif integrator == "SGNHT"
                phi = xi
            end
            p = O_step(p, h/occurences, phi, A, beta)
        elseif s == "D"
            mu = 1
            xi = D_step(p, h, xi, mu, beta)
        elseif s == "C"
            I = C_step(q, I, N, n, grad_U, t)
        end
    end

    return q, p
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
function run_simulation(q0, p0, Nsteps, h, integrator, steps, subset_prop, step_function, grad_U, data, A, beta, xi0, I0)
    q_traj = zeros(length(q0),Nsteps)
    p_traj = zeros(length(q0),Nsteps)
    t_traj = zeros(Nsteps)

    q = copy(q0)
    p = copy(p0)
    xi = copy(xi0)
    I = copy(I0)
    t = 0.0

    N = length(data)
    n = Int(round(N * subset_prop))

    for i in 1:Nsteps
        idx = randperm(N)[1:n]
        samples = data[idx]
        q, p = step_function(q, p, h, integrator, steps, N, n, A, beta, samples, grad_U, xi, I, t)

        t += h
        
        q_traj[:,i] = q
        p_traj[:,i] = p
        t_traj[i] = t
    end

    return q_traj, p_traj, t_traj
end