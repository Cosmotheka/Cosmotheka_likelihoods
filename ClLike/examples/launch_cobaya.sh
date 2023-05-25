#!/bin/bash
param=$1
nodes=$2
chains=$3
threads=$4
chains_per_node=$(($chains/$nodes))

export OMP_NUM_THREADS=$threads
export COBAYA_USE_FILE_LOCKING=false

/usr/local/shared/slurm/bin/srun -N $nodes -n $chains --ntasks-per-node $chains_per_node -m cyclic --mpi=pmi2 cobaya-run $param

# vim: tw=0
