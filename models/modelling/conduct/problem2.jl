loc = joinpath(@__DIR__, "..", "functions")
files = readdir(loc)
for file in files
    if endswith(file, ".jl")  # Check if the file is a Julia file
        include(joinpath(loc, file))
    end
end


using MLDatasets
using LinearAlgebra



# Load MNIST training data
train_x, train_y = MNIST.traindata()

# Flatten the images into feature vectors
train_x = reshape(train_x, :, size(train_x, 3))'

# Normalize the pixel values to the range [0, 1]
train_x = train_x / 255.0

# Number of features in the original dataset
n_features_original = size(train_x, 2)

# Generate a random projection matrix
n_components = 100
random_projection_matrix = randn(n_features_original, n_components)

# Project the features onto the random projection matrix
train_x_projected = train_x * random_projection_matrix



