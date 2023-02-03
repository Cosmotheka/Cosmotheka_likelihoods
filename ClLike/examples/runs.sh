#!/bin/bash

queue=cmb

# Likelihoods

############
addqueue -n 2x24 -s -q $queue -m 1  -c "desy1_3x2pt" -o log/desy1_3x2pt.out ./launch_cobaya.sh input/desy1_3x2pt.yml
