#!/bin/bash
#SBATCH --job-name=inf
#SBATCH --partition=sfscai
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
module purge
module load miniconda
source activate glm
python3 inference.py