#!/bin/bash
#SBATCH -e logs/cosolvkit_polar_random.job.err
#SBATCH -o logs/cosolvkit_polar_random.job.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16gb
#SBATCH --partition=alphafold

pwd; hostname; date
echo "Here: `pwd`"

# HARD CODED STUFF FOR ME
source /gpfs/home/nbruciaferri/.bashrc
conda activate cosolvkit

python /gpfs/home/nbruciaferri/cosolvkit/scripts/cosolvent_fill_the_void_setup.py -c /gpfs/home/nbruciaferri/validation/cosolvents_polar.json -f /gpfs/home/nbruciaferri/cosolvkit/forcefields.json -e amber -p /gpfs/home/nbruciaferri/validation/7jrn.pdb -o /gpfs/home/nbruciaferri/cosolvkit/results/polar_random -halton False
