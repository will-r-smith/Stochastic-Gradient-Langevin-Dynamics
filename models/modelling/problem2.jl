loc = joinpath(@__DIR__, "functions")
files = readdir(loc)
for file in files
    if endswith(file, ".jl")  # Check if the file is a Julia file
        include(joinpath(loc, file))
    end
end


using MLDatasets
using LinearAlgebra
using MultivariateStats
using Random
using Statistics
using LaTeXStrings
using Base.Threads
using JSON
using Plots


# Load MNIST training data
train_x, train_y = MNIST(split=:train)[:]
ind = findall(x -> x == 7 || x == 9, train_y)
y_train = train_y[ind]
y_train = ifelse.(y_train .== 7, 1, 0)
x_train = transpose(reshape(train_x, size(train_x, 1)*size(train_x, 2), size(train_x, 3))[:, ind])

# Normalize the pixel values to the range [0, 1]
x_train = x_train / 255.0


# Load MNIST test data
test_x, test_y = MNIST(split=:test)[:]
ind = findall(x -> x == 7 || x == 9, test_y)
y_test = test_y[ind]
y_test = ifelse.(y_test .== 7, 1, 0)
x_test = transpose(reshape(test_x, size(test_x, 1)*size(test_x, 2), size(test_x, 3))[:, ind])
# Normalize the pixel values to the range [0, 1]
x_test = x_test / 255.0




# Generate a random projection matrix
n_components = 100
# Perform Principal Component Analysis (PCA) to reduce dimensionality
pca_model = fit(PCA, Matrix(x_train'), maxoutdim=n_components)
x_train = MultivariateStats.transform(pca_model, Matrix(x_train'))

# Perform Principal Component Analysis (PCA) to reduce dimensionality
pca_model = fit(PCA, Matrix(x_test'), maxoutdim=n_components)
x_test = MultivariateStats.transform(pca_model, Matrix(x_test'))

train = hcat(x_train', y_train)
test = hcat(x_test', y_test)


prop_lst = 0.025:0.005:0.1
A_lst = 1:1:20
Nwalkers = 5
Nsteps = 100000
h = 0.001
beta = 1.0

#SGNHT - BADODAB
#SGLD - BAOAB
problem2_investigation(Nwalkers, prop_lst, A_lst, Nsteps, h, beta, "SGLD", "BAOAB", problem2_grad_U, n_components)


prop = 0.025
A = 10
Nsteps = 100000
h = 0.001
beta = 1.0
n_components = 100

#problem2_simulation("SGLD", "BAOAB", train, problem2_grad_U, Nsteps, h, A, beta, prop)