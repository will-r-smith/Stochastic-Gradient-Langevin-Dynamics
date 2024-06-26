#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N prob2_1_n1              
#$ -cwd                  
#$ -l h_rt=48:00:00 
#$ -l h_vmem=8G
#$ -pe interactivemem 12
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit: -l h_rt
#  memory limit: -l h_vmem
#  parallel environment: -pe interactivemem

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Julia
module load roslin/julia/1.9.0

# Run the program
julia -t 12 prob2_1_n1.jl

