#!/bin/bash
### Job name
#PS -N Charles_Test_Prime
### Mail to user
#PBS -m ae
#PBS -k o
#PBS -l nodes=4:ppn=16
#PBS -l walltime=240:00:00
#PBS -j oe

cd $PBS_O_WORKDIR
mpirun --bind-to-core ./VFS-Geophysics-3.1 > err                                                                
