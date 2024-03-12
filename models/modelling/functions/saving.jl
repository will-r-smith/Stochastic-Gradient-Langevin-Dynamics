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
function save_data()
    
end