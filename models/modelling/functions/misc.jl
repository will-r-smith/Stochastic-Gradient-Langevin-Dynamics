
function new_folder(model_type::AbstractString, problem, study_type, create_new)

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
        new_folder_path = joinpath(outputs_folder, new_folder_name)
        mkdir(new_folder_path)
    else
        # Find the highest model number among the matching folders
        max_model_number = maximum(parse(Int, match(r"\d+", folder).match) for folder in matching_folders)

        # Create a new folder name with the incremented model number
        if create_new == true
            new_folder_name = "$model_type"* "_" * "$(max_model_number + 1)"
            new_folder_path = joinpath(outputs_folder, new_folder_name)
            mkdir(new_folder_path)
        else
            new_folder_name = "$model_type"* "_" * "$(max_model_number)"
            new_folder_path = joinpath(outputs_folder, new_folder_name)
        end
    end

    return new_folder_path
end


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
function savedata(filename, data_dict)
    open(filename, "a") do io
        write(io, JSON.json(data_dict))
    end
end


# Define a function to load data from JSON files
function load_data_from_json(filename)
    json_data = JSON.parsefile(filename)
    return json_data
end
