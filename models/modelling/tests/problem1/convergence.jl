

function convergenceTest(q0, p0, Nsteps, h, A, beta, Samples, step_function, grad_U, N, n)
    # Run the simulation
    q_traj, p_traj, t_traj = run_simulation(q0, p0, Nsteps, h, A, beta, Samples, step_function, grad_U, N, n)

    l = length(q_traj[1,:])
    i = 100

    # Plot the running average
    conv_plot_1 = plot(1:i:l, cumsum(q_traj[1,:])[1:i:l] ./ (1:i:l), label="Running Average of q_traj", xlabel="Time", ylabel="Running Average", legend=:topright,linewidth = 2)
    conv_plot_2 = plot(1:i:l, cumsum(q_traj[2,:])[1:i:l] ./ (1:i:l), label="Running Average of q_traj", xlabel="Time", ylabel="Running Average", legend=:topright,linewidth = 2)

    layout = @layout([a; b])
    conv_plot = plot(conv_plot_1, conv_plot_2, layout=layout, legend=:topright)

    # Save the parameters used and q_traj data to a JSON file
    data_dict = Dict(
        "Nsteps" => Nsteps,
        "h" => h,
        "A" => A,
        "beta" => beta,
        "n" => n,
        "q_traj" => q_traj
    )

    return conv_plot, data_dict
end