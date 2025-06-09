#!/bin/bash
### Job name
#PS -N Charles_Test_Prime
### Mail to user
#PBS -m ae
#PBS -k o
#PBS -l nodes=1:ppn=1
#PBS -l walltime=24:00:00
#PBS -j oe

module load acml gnu openmpi

cd $PBS_O_WORKDIR
mpirun --bind-to-core ./data-3.2 -tis 0 -tie 0 -ts 100 > dataerr
