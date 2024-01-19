#!/bin/bash
#SBATCH -e openmm.job.err
#SBATCH -o openmm.job.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16gb
#SBATCH --time=71:59:59
#SBATCH --partition=alphafold

pwd; hostname; date
echo "Here: `pwd`"

# HARD CODED STUFF FOR ME
source /gpfs/home/nbruciaferri/.bashrc
conda activate cosolvkit

python /gpfs/home/nbruciaferri/cosolvkit/scripts/pipeline.py -c /gpfs/home/nbruciaferri/cosolvkit/cosolvents_lysozyme.json -f /gpfs/home/nbruciaferri/cosolvkit/forcefields.json -e amber -p /gpfs/home/nbruciaferri/cosolvkit/data/lysozyme/7l37.cif -o /gpfs/home/nbruciaferri/cosolvkit/results