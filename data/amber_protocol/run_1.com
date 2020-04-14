#!/bin/bash
#PBS -N cosolv_md
#PBS -e amber.err
#PBS -o amber.out
#PBS -m ae -M <email>
#PBS -l nodes=1:ppn=16:ngtx
#PBS -l walltime=200:00:00
#PBS -q gpu

cd ${PBS_O_WORKDIR}
module load amber/16
module load cuda

oldStep="mol"
step="1_wb-min"
pmemd.cuda -O -i ${step}.in -c ../scr/${oldStep}.inpcrd -ref ../scr/${oldStep}.inpcrd \
  -p ../scr/mol.prmtop -O -o ../out/${step}.out -inf ../out/${step}.info \
  -r ../scr/${step}.rst -x ../scr/${step}.nc -l ../out/${step}.log

oldStep="1_wb-min"
step="2_wb-heat"
pmemd.cuda -O -i ${step}.in -c ../scr/${oldStep}.rst -ref ../scr/${oldStep}.rst \
  -p ../scr/mol.prmtop -O -o ../out/${step}.out -inf ../out/${step}.info \
  -r ../scr/${step}.rst -x ../scr/${step}.nc -l ../out/${step}.log

oldStep="2_wb-heat"
step="3_wb-min"
pmemd.cuda -O -i ${step}.in -c ../scr/${oldStep}.rst -ref ../scr/${oldStep}.rst \
  -p ../scr/mol.prmtop -O -o ../out/${step}.out -inf ../out/${step}.info \
  -r ../scr/${step}.rst -x ../scr/${step}.nc -l ../out/${step}.log

oldStep="3_wb-min"
step="4_wb-heat"
pmemd.cuda -O -i ${step}.in -c ../scr/${oldStep}.rst -ref ../scr/${oldStep}.rst \
  -p ../scr/mol.prmtop -O -o ../out/${step}.out -inf ../out/${step}.info \
  -r ../scr/${step}.rst -x ../scr/${step}.nc -l ../out/${step}.log

oldStep="4_wb-heat"
step="5-min"
pmemd.cuda -O -i ${step}.in -c ../scr/${oldStep}.rst -ref ../scr/${oldStep}.rst \
  -p ../scr/mol.prmtop -O -o ../out/${step}.out -inf ../out/${step}.info \
  -r ../scr/${step}.rst -x ../scr/${step}.nc -l ../out/${step}.log

oldStep="5-min"
step="6-heat"
pmemd.cuda -O -i ${step}.in -c ../scr/${oldStep}.rst -ref ../scr/${oldStep}.rst \
  -p ../scr/mol.prmtop -O -o ../out/${step}.out -inf ../out/${step}.info \
  -r ../scr/${step}.rst -x ../scr/${step}.nc -l ../out/${step}.log

oldStep="6-heat"
step="7-equil"
pmemd.cuda -O -i ${step}.in -c ../scr/${oldStep}.rst -ref ../scr/${oldStep}.rst \
  -p ../scr/mol.prmtop -O -o ../out/${step}.out -inf ../out/${step}.info \
  -r ../scr/${step}.rst -x ../scr/${step}.nc -l ../out/${step}.log

oldStep="7-equil"
step="8-prod"
pmemd.cuda -O -i ${step}.in -c ../scr/${oldStep}.rst -ref ../scr/${oldStep}.rst \
  -p ../scr/mol.prmtop -O -o ../out/${step}.out -inf ../out/${step}.info \
  -r ../scr/${step}.rst -x ../scr/${step}.nc -l ../out/${step}.log

