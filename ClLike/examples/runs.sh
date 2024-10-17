#!/bin/bash
# Modify these three variables and add the chains that you want to
# launch/resume at the end of the script
queue=cmb
nodes=2
chains=4


######
# Automatic part
if [[ $queue == "cmb" ]]; then
    cores=24
elif  [[ $queue == "berg" ]]; then
    cores=28
elif  [[ $queue == "normal" ]]; then
    cores=1
else
    echo "queue not recognized. Choose cmb or berg"
    exit 1
fi

chains_per_node=$(($chains/$nodes))
threads=$(($cores/$chains_per_node))

if (( $chains % $nodes != 0 )); then
    echo "The number of chains or nodes is wrong"
    exit 1
elif (( $cores % $chains_per_node != 0 )); then
    echo "The number of chains per node ($chains_per_node) is not compatible with the number of cores ($cores) in the nodes"
    exit 1
fi


# Launcher
#########
create_launcher() {
    param=$1

    if [[ ! -d "./tmp" ]]; then
        mkdir ./tmp
    fi

    code=$(mktemp -p ./tmp)
    chmod +x $code
    cat <<EOF > $code
#!/bin/bash
export OMP_NUM_THREADS=$threads
export COBAYA_USE_FILE_LOCKING=false

/usr/local/shared/slurm/bin/srun -N $nodes -n $chains --ntasks-per-node $chains_per_node -m cyclic --mpi=pmi2 cobaya-run $param
EOF
    echo $code

}

add_job_to_queue() {
    param=$1
    logname=$2
    launcher=$(create_launcher $param)
    addqueue -n ${nodes}x${cores} -s -q $queue -m 1  -c $name -o log/$logname.log $launcher
}

launch_chain() {
    name=$1
    logname=$name
    param=input/$name.yml
    add_job_to_queue $param $logname
}

resume_chain() {
    name=$1
    param=input/$name/$name
    logname=${name}_resume
    add_job_to_queue $param $logname
}

minimize_chain() {
    name=$1
    logname=${name}_minimize
    param=input/$name.yml
    add_job_to_queue $param $logname
}

# Chains
############
# launch_chain desy1_3x2pt
# launch_chain desy1_3x2pt_onlybiases
# launch_chain FD_Garcia-Garcia2021
# launch_chain FD_Garcia-Garcia2021_dzMarg_mMarg

# Resume chain
# resume_chain desy1_3x2pt

# vim: tw=0
