using LaTeXStrings
using Base.Threads
using JSON
using Plots

function new_folder(model_type::AbstractString, problem, study_type)

    base_folder = @__DIR__  # Get the directory of the current file

    # Navigate up one directory
    parent_folder = joinpath(base_folder, "..", "..", "..")
    outputs_folder = joinpath(parent_folder, "outputs", problem, study_type)

    # Get a list of all subfolders in the base folder
    subfolders = filter(x -> isdir(joinpath(outputs_folder, x)), readdir(outputs_folder))

    # Filter subfolders based on the model type
    matching_folders = filter(folder -> startswith(folder, model_type), subfolders)

    if isempty(matching_folders)
        # If no matching model types are found, create a new folder starting at model 1
        new_folder_name = "$model_type" * "_1"
    else
        # Find the highest model number among the matching folders
        max_model_number = maximum(parse(Int, match(r"\d+", folder).match) for folder in matching_folders)

        # Create a new folder name with the incremented model number
        new_folder_name = "$model_type"* "_" * "$(max_model_number + 1)"
    end

    # Create the new folder
    new_folder_path = joinpath(outputs_folder, new_folder_name)
    mkdir(new_folder_path)

    return new_folder_path
end





function problem2_investigation(Nwalkers, prop_lst, A_lst, Nsteps, h, beta, integrator, steps, grad_U, n_components)

    function sigmoid(z)
        return 1.0 ./ (1.0 .+ exp.(-z))
    end
    
    function predict(features, weights)
        # Calculate the linear combination of features and weights
        z = features * weights
    
        # Apply the logistic function to get the probability
        probabilities = sigmoid(z)
        # Apply threshold (0.5) for binary classification
        return probabilities .>= 0.5
    end
    
    # Function to save data to a JSON file
    function save_data_to_json(filename, data_dict)
        JSON.open(filename, "a") do io
            JSON.print(io, data_dict)
        end
    end
    
    folder = new_folder(integrator, "problem_2", "investigations")
    filename = joinpath(folder, "data.json")

    accuracy_matrix = zeros(length(prop_lst), length(A_lst))
    
    lc = ReentrantLock()
    
    all_investigation_data = Dict()

    w_mean = zeros(n_components)
    
    # @sync @threads 
    for i in 1:length(prop_lst)
        println("i: ", i)
        # @threads 
        for j in 1:length(A_lst)
            println("  j: ", j)
            investigation_data = []
            
            @threads for k in 1:Nwalkers
                q0 = randn(n_components)
                p0 = randn(n_components)
                q_traj, p_traj, t_traj = run_simulation(q0, p0, Nsteps, h, integrator, steps, prop_lst[i], step_function, grad_U, train, A_lst[j], beta, 1, 1)
                
    
                lock(lc) do
                    w_mean = w_mean .+ mean(q_traj, dims=2)
                    # Predict the labels for the test set using the obtained parameters 'w'
                    ##predicted_labels = predict(x_test', w)

                    # Calculate accuracy
                    ##accuracy = sum(predicted_labels .== y_test) / length(y_test)
                    ##accuracy_matrix[i,j] += accuracy

                    #save_data_to_json(filename, Dict("Nwalkers" => repeats, "h" => h, "NSteps" => Nsteps, "n" => n_lst[i], "A" => A_lst[j], "repeat" => k, "data" => q_traj))
                    # Construct a dictionary with the data for this repetition
                    repetition_data = Dict(
                        "walker" => k,
                        "q_traj" => q_traj
                    )
                    push!(investigation_data, repetition_data)
                end
    
                # Store the investigation data in the all_investigation_data dictionary
                all_investigation_data[(i, j)] = Dict("proportion" => prop_lst[i], "friction" => A_lst[j], "repeats" => investigation_data)
            
            end

            w_mean = w_mean ./ Nwalkers
            # Predict the labels for the test set using the obtained parameters 'w'
            predicted_labels = predict(x_test', w_mean)

            # Calculate accuracy
            accuracy = sum(predicted_labels .== y_test) / length(y_test)
            accuracy_matrix[i,j] += accuracy
        end
    end

    
    all_investigation_data["params"] = Dict("NSteps" => Nsteps, "h" => h, "beta" => beta)
    
    #accuracy_matrix = accuracy_matrix ./ repeats
    
    contour(A_lst, prop_lst, accuracy_matrix, fill=true)
    contourf(A_lst, prop_lst, accuracy_matrix, levels=20, color=:turbo, colorbar=true)
    title!("Accuracy plot")
    # xaxis!(:log10)
    xlabel!("Friction")
    ylabel!("Subset Proportion")
    xticks!(A_lst)
    yticks!(prop_lst)
    
    savefig(folder * "/plot.png")
    
    save_data_to_json(joinpath(folder, "data.json"), all_investigation_data)
end






function problem2_simulation(integrator, steps, train, grad_U, Nsteps, h, A, beta, subset_prop)
    # Initialize one walker
    q0 = randn(size(train, 2)-1)
    p0 = randn(size(train, 2)-1)

    folder = new_folder(integrator, "problem_2", "simulations")
    
    # Run the simulation
    q_traj, p_traj, t_traj = run_simulation(q0, p0, Nsteps, h, integrator, steps, subset_prop, step_function, grad_U, train, A, beta, 1, 1)

    # Plot the running average
    plot(1:length(q_traj[1,:]), cumsum(q_traj[1,:]) ./ (1:length(q_traj[1,:])), label="Running Average of q_traj", xlabel="Time", ylabel="Running Average", legend=:topright)
    
    # Save the plot to the specified folder
    savefig(folder * "/plot.png")

    
    # Save the parameters used and q_traj data to a JSON file
    parameters_dict = Dict(
        "Nsteps" => Nsteps,
        "h" => h,
        "A" => A,
        "beta" => beta,
        "subset_prop" => subset_prop
    )
    data_dict = Dict(
        "parameters" => parameters_dict,
        "q_traj" => q_traj
    )
    json_filename = joinpath(folder, "simulation_data.json")
    JSON.open(json_filename, "w") do io
        JSON.print(io, data_dict)
    end
end



