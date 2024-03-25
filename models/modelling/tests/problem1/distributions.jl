
function distributionsTest(q0, p0, Nsteps, h, A, beta, Samples, step_function, grad_U, N, n_values)
    
    # Initialization to hold all density curves
    p_mu = plot(xlabel="μ", ylabel="Density", title="Comparisons of marginal distribution (density)", legend=:topright)
    p_gamma = plot(xlabel="γ", ylabel="Density", title="Comparisons of marginal distribution (density)", legend=:topright)

    # Calculate the parameters of the true distribution
    x_bar = sum(Samples) / N
    mu_N = x_bar * N / (N + 1)
    kappa_N = N + 1
    alpha_N = 1 + N / 2
    beta_N = 1 + sum((Samples .- x_bar).^2) / 2 + N * x_bar^2 / (2 * (N + 1))
    edge = [mu_N, mu_N, alpha_N/beta_N, alpha_N/beta_N]     # to plot theoretical density distribution

    v = 2 * alpha_N
    std_mu = sqrt(beta_N / (kappa_N * alpha_N))
    tdist = TDist(v)

    # # Run one long trajectory of Nsteps, using the BAOAB scheme
    # q_traj, p_traj, t_traj = run_simulation(q0, p0, Nsteps, h, A, beta, Samples, BAOAB_step, grad_gaussian, N, N);
    lc = ReentrantLock()

    q_traj_all = Dict()

    @sync @threads for n in n_values
        q_traj, _, _ = run_simulation(q0, p0, Nsteps, h, A, beta, Samples, step_function, grad_U, N, n)

        # x1 = range(minimum(q_traj[1,:]), maximum(q_traj[1,:]), length=100)
        # x2 = range(minimum(q_traj[2,:]), maximum(q_traj[2,:]), length=100)
    
        lock(lc) do
            # Generate the boundaries of the histogram
            edge[1] = minimum([minimum(q_traj[1,:]), edge[1]]); edge[2] = maximum([maximum(q_traj[1,:]), edge[2]])
            edge[3] = minimum([minimum(q_traj[2,:]), edge[3]]); edge[4] = maximum([maximum(q_traj[2,:]), edge[4]])

            stephist!(p_mu, q_traj[1,:], bins=800, normalize=:pdf, label="n = $n%", lw=1.5, alpha=0.5)
            stephist!(p_gamma, q_traj[2,:], bins=800, normalize=:pdf, label="n = $n%", lw=1.5, alpha=0.5)
            
            q_data = Dict(
                "n" => n,
                "q_traj" => q_traj
            )
            q_traj_all[("n_value = " * string(n))] = q_data
        end
    end

    # The theoretical density distribution
    x_standard_1 = range(edge[1], edge[2], length=100)
    x_standard_2 = range(edge[3], edge[4], length=100)
    q1_dist = pdf(mu_N .+ std_mu .* tdist, x_standard_1)
    q2_dist = pdf(Gamma(alpha_N, 1/beta_N), x_standard_2)

    plot!(p_mu, x_standard_1, q1_dist, color=:black, lw=1.5, label="True")
    plot!(p_gamma, x_standard_2, q2_dist, color=:black, lw=1.5, label="True")
    plot!(size=(620, 800)) 

    data_dict = Dict(
        "Nsteps" => Nsteps,
        "h" => h,
        "A" => A,
        "beta" => beta,
        "N" => N,
        "n_values" => n_values,
        "q_traj" => q_traj_all,
    )
    
    layout = @layout([a; b])
    combined_graph = plot(p_mu, p_gamma, layout=layout, legend=:topright)

    return combined_graph, data_dict

end