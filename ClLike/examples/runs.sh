#!/bin/bash

queue=cmb

# Likelihoods

############
# addqueue -n 2x24 -s -q $queue -m 1  -c "desy1_3x2pt" -o log/desy1_3x2pt.out ./launch_cobaya.sh input/desy1_3x2pt.yml
# addqueue -n 2x24 -s -q $queue -m 1  -c "desy1_3x2pt_onlybiases" -o log/desy1_3x2pt_onlybiases.out ./launch_cobaya.sh input/desy1_3x2pt_onlybiases.yml
addqueue -n 2x24 -s -q $queue -m 1  -c "FD" -o log/FD_Garcia-Garcia2021.out ./launch_cobaya.sh input/FD_Garcia-Garcia2021.yml
addqueue -n 2x24 -s -q $queue -m 1  -c "FD marg" -o log/FD_Garcia-Garcia2021_dzMarg_mMarg.out ./launch_cobaya.sh input/FD_Garcia-Garcia2021_dzMarg_mMarg.yml
