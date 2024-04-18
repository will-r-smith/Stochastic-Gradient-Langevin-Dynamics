# import Pkg; Pkg.add("MLDatasets")

using LaTeXStrings, Random, Plots, Serialization, StatsBase, Base.Threads, LinearAlgebra, Distributions, PlotUtils

function A_step(qp, h)
    q, p = qp
    q = q + h * p
    return [q, p]
end

function B_step(qp, h, force)
    q, p = qp
    F = force
    p = p + h * F
    return [q, p]
end

function O_step(qp, h, A, beta)
    q, p = qp
    alpha = exp(-h * A)
    R = randn(length(q))
    p = alpha * p + sqrt(1 / beta) * sqrt(1 - exp(-2 * h * A)) * R
    return [q, p]
end

function BAOAB_step(q, p, h, A, beta, force)
    qp = copy([q, p])
    qp = B_step(qp, h/2, force)
    qp = A_step(qp, h/2)
    qp = O_step(qp, h, A, beta)
    qp = A_step(qp, h/2)
    qp = B_step(qp, h/2, force)
    q, p = qp
    return q, p
end

function grad_BLR(data, w, N, n)
    # 2nd model - Large Scale Bayesian Logistic Regression
    dim = size(data, 2) - 1
    x = data[:, 1:dim]
    y = data[:, end]

    sum = zeros(dim)
    for i in 1:n
        a = exp(- y[i] * dot(w, x[i, :]))
        sum = sum .+ ((-y[i] * a  / (1 + a)) .* x[i, :])
    end

    w += sum .* N/n

    return w
end

function run_simulation(q0, p0, Nsteps, h, A, beta, Samples, step_function, grad_U, N, n)
    dim = size(Samples, 2) - 1
    q_traj = zeros(dim,Nsteps)
    p_traj = zeros(dim,Nsteps)
    t_traj = zeros(Nsteps)

    q = copy(q0)
    p = copy(p0)
    t = 0.0

    for i in 1:Nsteps
        idx = randperm(N)[1:n]
        data = Samples[idx,:]
        force = - grad_U(data, q, N, n)
        q, p = step_function(q, p, h, A, beta, force)
        t += h
        
        q_traj[:,i] = q
        p_traj[:,i] = p
        t_traj[i] = t
    end

    return q_traj, p_traj, t_traj
end

function sigmoid(z)
    return 1.0 ./ (1.0 .+ exp.(-z))
end

function predict_data(features, weights)
    # Calculate the linear combination of features and weights
    z = first(features' * weights)
    # Apply the logistic function to get the probability
    probabilities = sigmoid(z)
    # Apply threshold (0.5) for binary classification
    predictions = ifelse.(probabilities .>= 0.5, 1, -1)
    
    return predictions
end

function predict_BLR(q_mean, x_test, y_test)
    t = sum(q_mean)
    if isnan(t) || isinf(t)
        return 0
    end
    predicted_labels = zeros(size(x_test, 1))
    for i in 1:size(x_test, 1)
        predicted_labels[i] = predict_data(x_test[i,:], q_mean)
    end
    accuracy = sum(predicted_labels .== y_test) / length(y_test)
    return accuracy
end

function run_predict(q0, p0, Nsteps, h, A, beta, Samples, step_function, grad_U, N, n, x_test, y_test)
    dim = size(Samples, 2) - 1
    # q_traj = zeros(dim,Nsteps)
    # p_traj = zeros(dim,Nsteps)
    # t_traj = zeros(Nsteps)

    q = copy(q0)
    q_sum = zeros(dim)
    p = copy(p0)
    t = 0.0

    for i in 1:Nsteps
        idx = randperm(N)[1:n]
        data = Samples[idx,:]
        force = - grad_U(data, q, N, n)
        q, p = step_function(q, p, h, A, beta, force)
        t += h
        q_sum += q
    end
    q_mean = q_sum ./ Nsteps

    t = sum(q_mean)
    if isnan(t) || isinf(t)
        return 0
    end
    predicted_labels = zeros(size(x_test, 1))
    for i in 1:size(x_test, 1)
        predicted_labels[i] = predict_data(x_test[i,:], q_mean)
    end
    accuracy = sum(predicted_labels .== y_test) / length(y_test)

    return accuracy
end

# function RMSE_postmean_BLR(q_mean, x_test, y_test)
#     if isnan(q_mean) || isinf(q_mean)
#         return 0
#     end
# end

function random_projection_matrix(rows, cols)
    matrix = zeros(Int, rows, cols)
    for i in 1:rows
        for j in 1:cols
            rand_num = rand()
            if rand_num <= 1/6
                matrix[i, j] = 1
            elseif rand_num <= 2/6
                matrix[i, j] = -1
            end
        end
    end
    matrix = matrix .* sqrt(3/cols)
    # matrix = rand(Normal(0, 1 / sqrt(rows)), rows, cols)
    return matrix 
end

function save_variable(variable, file_name)
    name = string(file_name, ".jls")
    open(name, "w") do file
        serialize(file, variable)
    end
end

using MLDatasets

dim = 100
train_x, train_y = MNIST(split=:train)[:]
ind = findall(x -> x == 7 || x == 9, train_y)
Random.seed!(42)
rpm = random_projection_matrix(size(train_x, 1)*size(train_x, 2), dim)

y_train = train_y[ind]
y_train = ifelse.(y_train .== 7, 1, -1)
x_train = transpose(reshape(train_x, size(train_x, 1)*size(train_x, 2), size(train_x, 3))[:, ind])
x_train = x_train * rpm
train = hcat(x_train, y_train)

test_x, test_y = MNIST(split=:test)[:]
ind = findall(x -> x == 7 || x == 9, test_y)
y_test = test_y[ind]
y_test = ifelse.(y_test .== 7, 1, -1)
x_test = transpose(reshape(test_x, size(test_x, 1)*size(test_x, 2), size(test_x, 3))[:, ind]);
x_test = x_test * rpm;

# Nsteps = 10^4, repeat 6 times for approx 213 min - Bohan

n_lst = collect(1:50)
A_lst = [0.1, 1, 10, 100]
# A_lst = 101:-4:1
predict_matrix_perc = zeros(length(A_lst), length(n_lst));
# RMSE_postmean = zeros(length(n_lst), length(A_lst));
repeats = 12

# Initialize one walker
dim = size(train, 2) - 1
q0 = randn(dim); p0 = randn(dim)
Nsteps = 100000; h = 0.001; beta = 1.0; N = size(train, 1)

# calculate the real distribution


lc = ReentrantLock()

# @sync @threads 
for i in 1:length(A_lst)
    A = A_lst[i]
    # @threads 
    for j in 1:length(n_lst)
        accuracy = 0.0
        # RMSE = 0.0
        n = n_lst[j]
        @sync @threads for k in 1:repeats
            acc = run_predict(q0, p0, Nsteps, h, A, beta, train, BAOAB_step, grad_BLR, N, n, x_test, y_test)
            lock(lc) do
                accuracy += acc
            end
        end
        predict_matrix_perc[i,j] = accuracy / repeats
        # RMSE_postmean[i,j] = RMSE / repeats
    end
end

save_variable(predict_matrix_perc, "predict_matrix_perc");
# save_variable(RMSE_postmean, "RMSE_postmean");
println("fin")