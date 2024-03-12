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
function plot_density(data, N, q_traj, folder)
    x_bar = sum(data)/N
    mu_N = x_bar * N / (N+1);
    kappa_N = N+1;
    alpha_N = 1 + N/2;
    beta_N = 1 + sum((data .- x_bar).^2) / 2 + N * x_bar^2 / (2 * (1+N));
    
    x1 = range(minimum(q_traj[1,:]), stop=maximum(q_traj[1,:]), length=100)
    x2 = range(minimum(q_traj[2,:]), stop=maximum(q_traj[2,:]), length=100)
    
    v = 2 * alpha_N
    stnd = sqrt(beta_N / (kappa_N*alpha_N))
    tdist = TDist(v)
    q1_dist = pdf(mu_N .+ stnd .* tdist, x1)
    q2_dist = pdf(Gamma(alpha_N, 1/beta_N), x2)
    
    q1_plot = stephist(q_traj[1,:], bins=1000, normalize=true, label="BAOAB", xlabel="μ", ylabel="Density", color="blue")
    q2_plot = stephist(q_traj[2,:], bins=1000, normalize=true, label="BAOAB", xlabel="γ", ylabel="Density", color="blue")
    plot!(q1_plot, x1, q1_dist, color="red", lw=2, label="True")
    plot!(q2_plot, x2, q2_dist, color="red", lw=2, label="True")
    
    plot(q1_plot, q2_plot, layout=(1,2), size=(1000, 450))

    savefig(folder * "/densities.png")

end


using Plots


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
function plot_bias_friction(q_true, A_values, q_traj, N_steps, folder)
    bias = zeros(length(A_values), length(q_true))
    for i in 1:length(A_values)
        for j in 1:length(q_true)
            bias[i, j] = abs(sum(q_traj[i, j, :])/N_steps - q_true[j])
        end
    end

    plot(A_values, bias[:, 1],
         xlabel="A_values", ylabel="Bias", title="Bias vs A_values",
         lw=2)
    savefig(folder * "/friction_plot.png")

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
function plot_subset_accuracy(q_true, subset_values, q_traj, N_steps, folder)
    accuracy = zeros(length(subset_values), length(q_true))
    for i in 1:length(subset_values)
        for j in 1:length(q_true)
            accuracy[i, j] = abs(sum(q_traj[i, j, :])/N_steps - q_true[j])
        end
    end

    plot(subset_values, accuracy[:, 1],
         xlabel="subset proportion", ylabel="Bias", title="Bias vs Subset proportion",
         lw=2)
    savefig(folder * "/subset_plot.png")

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
function plot_convergence(data, q_traj, folder)
    mu_samples = q_traj[1, :]
    gamma_samples = q_traj[2, :]

    # Plot running averages of observables
    p1 = plot(
        1:length(mu_samples), cumsum(mu_samples) ./ (1:length(mu_samples)),
        label="μ", xlabel="iteration t", ylabel="Running Average", legend=:topright,
        title="Running Average of μ"
    )

    # Plot horizontal line for reference on the first subplot
    plot!(p1, [1, length(mu_samples)], [mean(data), mean(data)], linestyle=:dash, linewidth=1, label="Mean of Samples")

    # Plot running averages of γ⁻¹
    p2 = plot(
        1:length(gamma_samples), cumsum(gamma_samples) ./ (1:length(gamma_samples)),
        label="γ⁻¹", xlabel="iteration t", ylabel="Running Average", legend=:bottomright,
        title="Running Average of γ⁻¹"
    )

    # Plot horizontal line for reference on the second subplot
    plot!(p2, [1, length(gamma_samples)], [1/var(data), 1/var(data)], linestyle=:dash, linewidth=1, label="1/Std of Samples")

    # Display the plots side by side
    plot(p1, p2, layout=(1, 2), size=(800, 400))
    savefig(folder * "/covergence_plot.png")
    
end