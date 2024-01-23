#!/bin/bash
#SBATCH -e logs/cosolvkit_non_polar_far.job.err
#SBATCH -o logs/cosolvkit_non_polar_far.job.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16gb
#SBATCH --partition=alphafold

pwd; hostname; date
echo "Here: `pwd`"

# HARD CODED STUFF FOR ME
source /gpfs/home/nbruciaferri/.bashrc
conda activate cosolvkit

python /gpfs/home/nbruciaferri/cosolvkit/scripts/cosolvent_find_and_replace_setup.py -c /gpfs/home/nbruciaferri/validation/cosolvents_non_polar.json -r /gpfs/home/nbruciaferri/validation/7jrn.pdb -o /gpfs/home/nbruciaferri/cosolvkit/results/non_polar_far