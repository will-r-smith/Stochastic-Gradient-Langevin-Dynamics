function bias_pdf(q_traj, mu_dist, gamma_dist)
    x1 = range(minimum(q_traj[1,:]), maximum(q_traj[1,:]), length=100)
    x2 = range(minimum(q_traj[2,:]), maximum(q_traj[2,:]), length=100)
    
    edge_q1 = collect(x1)
    edge_q2 = collect(x2)

    hist_q1 = fit(Histogram, q_traj[1,:], edge_q1)
    hist_q2 = fit(Histogram, q_traj[2,:], edge_q2)

    bin_centers_q1 = (edge_q1[1:end-1] .+ edge_q1[2:end]) ./ 2
    bin_centers_q2 = (edge_q2[1:end-1] .+ edge_q2[2:end]) ./ 2

    bin_widths_q1 = diff(edge_q1)
    total_counts_q1 = sum(hist_q1.weights)
    hist_density_q1 = hist_q1.weights ./ (total_counts_q1 .* bin_widths_q1)

    bin_widths_q2 = diff(edge_q2)
    total_counts_q2 = sum(hist_q2.weights)
    hist_density_q2 = hist_q2.weights ./ (total_counts_q2 .* bin_widths_q2)

    q1_dist = pdf(mu_dist, bin_centers_q1)
    q2_dist = pdf(gamma_dist, bin_centers_q2)

    errors_q1 = q1_dist - hist_density_q1
    errors_q2 = q2_dist - hist_density_q2

    # Calculate mean square error (MSE) and root mean square error (RMSE)
    mse_q1 = mean(errors_q1.^2)
    rmse_q1 = sqrt(mse_q1)
    mse_q2 = mean(errors_q2.^2)
    rmse_q2 = sqrt(mse_q2)
    return [rmse_q1, rmse_q2]
end


function biasTest(n_lst, A_lst, h, beta, N, repeats, Samples, grad_U, save_trajectories)
    
    bias_matrix1 = zeros(length(n_lst), length(A_lst))
    bias_matrix2 = zeros(length(n_lst), length(A_lst))
    
    # calculate the real distribution
    x_bar = sum(Samples)/N; 
    mu_N = x_bar * N / (N+1); kappa_N = N+1; alpha_N = 1 + N/2; beta_N = 1 + sum((Samples .- x_bar).^2) / 2 + N * x_bar^2 / (2 * (1+N));
    v = 2 * alpha_N; std_mu = sqrt(beta_N / (kappa_N*alpha_N)); tdist = TDist(v)
    mu_dist = mu_N .+ std_mu .* tdist
    gamma_dist = Gamma(alpha_N, 1/beta_N)

    lc = ReentrantLock()

    all_investigation_data = Dict()

    # @sync @threads 
    for i in 1:length(n_lst)
        # @threads 
        for j in 1:length(A_lst)
            investigation_data = Dict()
            bias1, bias2 = zeros(2)
            @threads for k in 1:repeats
                Nsteps = Int(500 * A_lst[j] / h)
                
                q_traj, p_traj, t_traj = run_simulation(q0, p0, Nsteps, h, A_lst[j], beta, Samples, BAOAB_step, grad_U, N, n_lst[i])

                lock(lc) do
                    bias1, bias2 = bias_pdf(q_traj, mu_dist, gamma_dist) .+ [bias1, bias2]

                    if save_trajectories == true
                        repetition_data = Dict(
                            "walker" => k,
                            "q_traj" => q_traj
                        )
                        push!(investigation_data, repetition_data)

                    end
                end
            end


            bias_matrix1[i,j], bias_matrix2[i,j] = [bias1, bias2]

            if save_trajectories == true
                all_investigation_data[(n_lst[i], A_lst[j])] = Dict("proportion" => n_lst[i], "friction" => A_lst[j], "bias" => [bias1, bias2], "repeats" => investigation_data)
            else
                all_investigation_data[(n_lst[i], A_lst[j])] = Dict("proportion" => n_lst[i], "friction" => A_lst[j], "bias" => [bias1, bias2])
            end

        end

    end

    contour(A_lst, n_lst, bias_matrix1, fill=true)
    biasplot1 = contourf(A_lst, n_lst, bias_matrix1, levels=20, color=:turbo, colorbar=true, title = "Bias plot", xlabel="friction", ylabel="percentage", xticks = sort(A_lst)[1:2:end], yticks = sort(n_lst)[1:5:end])
    biasplot2 = contourf(A_lst, n_lst, bias_matrix2, levels=20, color=:turbo, colorbar=true, title = "Bias plot", xlabel="friction", ylabel="percentage", xticks = sort(A_lst)[1:2:end], yticks = sort(n_lst)[1:5:end])
    

    return biasplot1, biasplot2, all_investigation_data
end



function collatePlot(json_dir, parameter)

    # Define a dictionary to hold all the combined data
    all_investigation_data = Dict{Tuple{Int, Int}, Dict}()

    # Loop through each JSON file in the directory
    for file in readdir(json_dir)

        data = load_data_from_json(joinpath(json_dir, file))

        for item in data
            # Extract the proportion, friction, and bias data from the loaded dictionary
            proportion = item.second["proportion"]
            friction = item.second["friction"]
            bias = item.second["bias"][parameter]  # Extracting the first element of bias
            


            # Create a key for the combined dictionary
            key = (proportion, friction)
            
            # Update the combined dictionary with the extracted data
            all_investigation_data[key] = Dict("proportion" => proportion, "friction" => friction, "bias" => bias)
        end
    end

    # Extract data for plotting
    x_values = [key[2] for key in keys(all_investigation_data)]  # Friction values
    y_values = [data["proportion"] for data in values(all_investigation_data)]  # Proportion values
    z_values = [data["bias"] for data in values(all_investigation_data)]  # Bias values
    """
    # Sort x_values and y_values
    sorted_indices_x = sortperm(x_values)
    x_values = x_values[sorted_indices_x]
    y_values = y_values[sorted_indices_x]
    z_values = z_values[sorted_indices_x]


    sorted_indices_y = sortperm(y_values)
    x_values = x_values[sorted_indices_y]
    y_values = y_values[sorted_indices_y]
    z_values = z_values[sorted_indices_y]
    
    println(y_values)
    y_values = unique(y_values)
    x_values = unique(x_values)

    println(y_values)

    println("A values:")
    println(x_values)
    println("n values:")
    println(y_values)
    
    # Reshape z_values to match dimensions of x_values and y_values
    z_values = reshape(z_values, length(y_values), length(x_values))
    y_sort = sortperm(y_values)
    x_sort = sortperm(x_values)

    x_values = x_values[x_sort]
    y_values = y_values[y_sort]
    z_values = z_values[y_sort, :]
    z_values = z_values[:, x_sort]
    """

    # Combine x, y, and z vectors into a DataFrame
    df = DataFrame(x = x_values, y = y_values, z = z_values)

    # Sort the DataFrame based on x and y values
    sorted_df = sort(df, [:x, :y])

    # Separate the sorted DataFrame into sorted x, y, and z vectors
    sorted_x_positions = sorted_df.x
    sorted_y_positions = sorted_df.y
    sorted_z_positions = sorted_df.z

    # Find unique x and y positions
    unique_x_positions = unique(sorted_x_positions)
    unique_y_positions = unique(sorted_y_positions)

    # Initialize the 2D array for z values
    z_array = zeros(length(unique_y_positions), length(unique_x_positions))

    # Fill the z_array with sorted z values
    for i in 1:nrow(sorted_df)
        x_index = findfirst(x -> x == sorted_df[i, :x], unique_x_positions)
        y_index = findfirst(y -> y == sorted_df[i, :y], unique_y_positions)
        z_array[y_index, x_index] = sorted_df[i, :z]
    end



    biasPlot = contourf(unique_x_positions, unique_y_positions, z_array, levels=20, color=:turbo, colorbar=true, xlabel="Friction", ylabel="Proportion", title="Bias Plot")

    return biasPlot

end


function collectBiasData(folder, path, h, beta, N, repeats, Samples, grad_U, min_n, max_n, interval_n, sets_n, min_A, max_A, interval_A, sets_A)
    f = joinpath(folder, path)
    files = readdir(f)
    # Check if files array is empty
    if isempty(files)
        set = 0
    else
        # Extract numeric parts from filenames and convert to integers
        set_numbers = parse.(Int, replace.(files, r"set(\d+)\.json" => s"\1"))
        # Find the maximum set number
        set = maximum(set_numbers)
    end
    for s_n in 1:sets_n
        for s_A in 1:sets_A
            set += 1
            l_n = min_n+div(max_n-min_n+1, sets_n)*(s_n-1)
            u_n = min_n+div(max_n-min_n+1, sets_n)*s_n-1
    
            n_lst = l_n:interval_n:u_n
    
            l_A = min_A+div(max_A-min_A+1, sets_A)*(s_A-1)
            u_A = min_A+ div(max_A-min_A+1, sets_A)*s_A-1
    
            A_lst = l_A:interval_A:u_A
    
    
            testPlot1, testPlot2, dataDict = biasTest(n_lst, A_lst, h, beta, N, repeats, Samples, grad_U, false)

            biasDataFilename = joinpath(f, "set"*string(set)*".json")
            savedata(biasDataFilename, dataDict)
        end
    end
    
end