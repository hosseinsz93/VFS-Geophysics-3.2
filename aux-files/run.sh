#!/bin/bash
### Job name
#PBS -N Charles_Test_Prime
### Mail to user
#PBS -m ae
#PBS -k o
#PBS -l nodes=4:ppn=16
#PBS -l walltime=240:00:00
#PBS -j oe

module load acml gnu openmpi

cd $PBS_O_WORKDIR
mpirun --bind-to-core ./VFS-Geophysics-3.2 >> err
