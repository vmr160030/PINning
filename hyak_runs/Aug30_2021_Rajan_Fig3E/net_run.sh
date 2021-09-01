#!/bin/bash

## Job Name

#SBATCH --job-name=PINning

## Allocation Definition

## The account and partition options should be the same except in a few cases (e.g. ckpt queue and genpool queue).

#SBATCH --account=stf
#SBATCH --partition=stf

## Resources

## Total number of Nodes

#SBATCH --nodes=1   

## Number of cores per node

#SBATCH --ntasks-per-node=28

## Walltime (3 hours). Do not specify a walltime substantially more than your job needs.

#SBATCH --time=12:00:00

## Memory per node. It is important to specify the memory since the default memory is very small.

## For mox, --mem may be more than 100G depending on the memory of your nodes.

## For ikt, --mem may be 58G or more depending on the memory of your nodes.

## See above section on "Specifying memory" for choices for --mem.

#SBATCH --mem=50G

## Specify the working directory for this job

#SBATCH --chdir=/usr/lusers/vyomr/Fairhall_code/PINning/hyak_runs/Aug30_2021_Rajan_Fig3E/

##turn on e-mail notification

#SBATCH --mail-type=ALL

#SBATCH --mail-user=vyomr@uw.edu

## export all your environment variables to the batch job session

#SBATCH --export=all
module load contrib/anaconda-2019.03
conda init
conda activate neuro
source activate neuro
python Aug30_Rajan_Fig3E.py
