#!/bin/bash
#SBATCH --job-name=comb_t
#SBATCH --partition=debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=10:00:00
module purge
module load miniconda
source activate glm
python combine_data.py