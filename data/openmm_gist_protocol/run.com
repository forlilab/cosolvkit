#!/bin/bash
#PBS -N gist
#PBS -e openmm.err
#PBS -o openmm.out
#PBS -m ae -M <email_address>
#PBS -l nodes=1:ppn=1:ngtx
#PBS -l walltime=200:00:00
#PBS -q gpu

cd ${PBS_O_WORKDIR}
module load openmm/7.4.2
module load cuda/10.0

python run_md.py
cpptraj -i merge.inp > merge.out
cpptraj -i gist.inp > gist.out
