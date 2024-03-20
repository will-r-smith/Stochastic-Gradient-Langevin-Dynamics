# Stochastic-Gradient-Langevin-Dynamics
Investigation into the use of kinetic Langevin dynamics in the setting of Bayesian sampling for machine learning.

### To Do:

- [ ] Write a function that runs the model, plots the graphs and saves the data and outputs to a new file in the outputs folder 

    - [x] Write the functions for the steps of the standard Stochastic Gradient Langevin Dynamics in a somewhat generalised fashion

    - [x] Write the force function

    - [x] Write the code for the model function (e.g. BAOAB) based off a character string

    - [x] Write the code to plot the figures required

    - [x] Write the code to save a new folder based off existing models and the name of the model and its important parameters

    - [ ] Write the code to populate the new folder with a csv file containing the data, a json file containing the model parameters and the date and model type etc. and also save the figures to the folder

        - [x] Save figures

        - [ ] Save data

        - [ ] Save parameters in json

- [ ] Create Models

    - [ ] Model 1 (Example 1 in CCAdL)

        - [x] Derive force function

        - [x] Write up force function derivation in paper

        - [x] Write force function code

        - [x] Generate data

        - [x] Run model

    - [ ] Model 2 (Example 2 in CCAdL)

        - [ ] Derive force function

        - [ ] Write up force function derivation in paper

        - [ ] Write force function code

        - [ ] Download and save dataset to the data folder

        - [ ] Run model

        - [ ] Run model with Stochastic Gradient Langevin Dynamics

    - [ ] Model 3 (Create our own machine learning problem)

        - [ ] Derive force function (perhaps use the sigmoid classification problem in 4.2 but with a new dataset)

        - [ ] Write up force function derivation in paper

        - [ ] Write force function code

        - [ ] Download and save dataset to the data folder

        - [ ] Run model with Stochastic Gradient Langevin Dynamics

- [ ] Apply other integrators

    - [ ] CCAdL

        - [ ] Derive the O-step

        - [ ] Write up derivation in the paper

        - [ ] Write the code for the steps

        - [ ] Run experiments using model 3

    - [x] SGNHT

        - [x] Derive the O-step

        - [ ] Write up derivation in the paper

        - [x] Write the code for the steps

        - [x] Run experiments using model 3






#### Existing Models:

- Stochastic Gradient Langevin Dynamics SGLD [1]

- Modified Stochastic Gradient Langevin Dynamics mSGLD [2]

- Stochastic Gradient Hamiltonian Monte Carlo SGHMC [3]

- Stochastic Gradient Nose-Hoover Thermostat SGNHT [4]

- Adaptive Langevin thermostat  Ad-Langevin [5]

- Covariance-Controlled Adaptive Langevin CCAdL [6]




- [ ] subset proportion vs friction parameter plot with different bias (colourbar) (or accuracy with problem 2)