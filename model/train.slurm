#!/bin/bash
#SBATCH --job-name=avg_lm
#SBATCH --partition=sfscai
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mem=200G
module purge
module load miniconda
source activate glm
python -m train.py