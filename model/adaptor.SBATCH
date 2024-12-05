#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=5:00:00
#SBATCH --mem=80GB
#SBATCH --job-name=adaptor
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=qz2086@nyu.edu

module purge

singularity exec --nv \
	    --overlay /scratch/qz2086/singularity4/overlay-25GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
	    /bin/bash -c "source /ext3/env.sh; python test.py"