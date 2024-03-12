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
function investigation(problem, q0, p0, q_true, Nsteps, N, h, regime, subset_values, step_function, grad_U, data, A, beta)

    integrator = collect(split(regime, "_"))[1]
    steps = collect(split(collect(split(regime, "_"))[2], ""))

    folder = new_folder(integrator, problem, "investigations")

    q_traj_A = zeros(length(A_values), length(q0), Nsteps)
    p_traj_A = zeros(length(A_values), length(q0), Nsteps)
    t_traj_A = zeros(length(A_values), Nsteps)

    for i in 1:length(A_values)
        q_t, p_t, t_t = run_simulation(q0, p0, Nsteps, h, integrator, steps, 0.1, step_function, grad_U, data, A_values[i], beta, A_values[i], I(length(q0)))
        q_traj_A[i, :, :] = q_t
        t_traj_A[i, :] = t_t
    end

    plot_bias_friction(q_true, A_values, q_traj_A, Nsteps, folder)

    q_traj_S = zeros(length(subset_values), length(q0), Nsteps)
    p_traj_S = zeros(length(subset_values), length(q0), Nsteps)
    t_traj_S = zeros(length(subset_values), Nsteps)

    for i in 1:length(subset_values)
        q_t, p_t, t_t = run_simulation(q0, p0, Nsteps, h, integrator, steps, subset_values[i], step_function, grad_U, data, 10, beta, 10, I(length(q0)))
        q_traj_S[i, :, :] = q_t
        t_traj_S[i, :] = t_t
    end

    plot_subset_accuracy(q_true, subset_values, q_traj_S, Nsteps, folder)

end


function simulation(problem, q0, p0, q_true, Nsteps, N, h, regime, subset, step_function, grad_U, data, A, beta)
    if subset > 1
        subset = length(data) / subset
    end

    integrator = collect(split(regime, "_"))[1]
    steps = collect(split(collect(split(regime, "_"))[2], ""))

    q_traj, p_traj, t_traj = run_simulation(q0, p0, Nsteps, h, integrator, steps, 0.1, step_function, grad_U, data, A, beta, A, I(length(q0)))

    folder = new_folder(integrator, problem, "simulations")

    plot_density(data, N, q_traj, folder)

    plot_convergence(data, q_traj, folder)

    return q_traj, p_traj, t_traj
end