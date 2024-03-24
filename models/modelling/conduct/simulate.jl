loc = joinpath(@__DIR__, "..", "functions")
files = readdir(loc)
for file in files
    if endswith(file, ".jl")  # Check if the file is a Julia file
        include(joinpath(loc, file))
    end
end




using Plots, Distributions, LinearAlgebra, Random

# Initialize one walker
q0 = [0,1]      # just for initialization
p0 = randn(2)

Nsteps = 1000000
h = 0.01
A = 50
beta = 1.0
N = 100
subset_prop = 0.01
data = randn(N)
q_true = [0, 1]

q_traj, p_traj, t_traj = simulation("problem_1", q0, p0, q_true, Nsteps, N, h, "SGNHT_BADODAB", subset_prop, step_function, model1_grad_U, data, A, beta)
q_traj, p_traj, t_traj = simulation("problem_1", q0, p0, q_true, Nsteps, N, h, "SGLD_BAOAB", subset_prop, step_function, model1_grad_U, data, A, beta)