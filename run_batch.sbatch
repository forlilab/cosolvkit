#!/bin/bash
#SBATCH --job-name=cosolvkit_setup
#SBATCH --output=cosolvkit_setup.o
#SBATCH --error=cosolvkit_setup.e
#SBATCH --partition=highmem
#SBATCH --time=71:59:59
#SBATCH --exclusive

bash run.sh
