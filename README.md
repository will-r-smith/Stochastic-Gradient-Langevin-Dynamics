# Stochastic-Gradient-Langevin-Dynamics
Investigation into the use of kinetic Langevin dynamics in the setting of Bayesian sampling for machine learning.


## Introduction

As a gradient-based Monte Carlo sampling algorithm, Langevin dynamics has applications in Bayesian inference. In the setting of Bayesian machine learning, datasets are used to define a differentiable objective function. From this an effective friction law is obtained which is applied in the framework of Langevin dynamics. However, machine learning applications frequently deal with very large, high-dimensional datasets and as such direct sampling with the entire dataset is often computationally infeasible. In order to mitigate this issue, the model is often combined with characteristics of stochastic gradient descent methods. Stochastic gradient methods use a subset of the data at each timestep to approximate the true gradient,
$$
        w_{t+1} = w_t - \frac{N}{n} \sum_{k=1}^{n} \nabla U(w_t, q_k).
$$
The principle is that the expectation of the stochastic gradient is equivalent to that of the entire dataset,
$$
        \mathbb{E}\left[ \frac{N}{n} \sum_{k=1}^{n} \nabla U(w_t, q_k) \right] = \sum_{i=1}^{N} \nabla U(w_t,q_i),
$$
and as such serves as an appropriate approximation. This stochastic gradient estimator forms the likelihood gradient terms in the stochastic gradient Langevin dynamics (SGLD). SGLD can be seen as Langevin dynamics applied to posterior distributions where samples are taken from a posterior distribution of parameters arising from subsets of the data.

Langevin dynamics is an extension of Brownian dynamics that incorporates inertial effects. The dynamics involve the introduction of a friction parameter. In kinetic Langevin dynamics, the system of stochastic differential equations for a particle is
$$
    dq &= pdt \\ 
    dp &= -\nabla U(q)dt - A pdt + \sqrt{2 A \beta^{-1}} dW_t 
$$

where $q$ and $p$ are position and momentum, respectively, $\beta^{-1}$ is the inverse temperature and $A$ is the friction parameter. The limiting dynamical case, as $A \rightarrow \infty$, corresponds to the purely diffusive Brownian regime.

Whilst stochastic gradients can offer significant reductions in computational cost, the smaller subsets are associated with greater variance and one must be mindful of bias introduced into the model through the approach. The effect of the subset size on bias has been well documented but the effect of the friction parameter on bias has been studied less. \cite{shang2015covariance} investigated bias for several other gradient-based Monte Carlo sampling algorithms including SGNHT and SGHMC but only considered two values for the friction parameter. This report seeks to investigate in detail the effect of the friction parameter on the bias introduced into the model as a result of the stochastic gradients, specifically for SGLD. 

Our approach is to conduct investigations into the induced bias for two experimental applications, performing simulations for an array of friction parameters and subset proportions. 
