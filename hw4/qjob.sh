#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=10:00:00
#PBS -l mem=30GB
#PBS -N pinesol_rnn
#PBS -M alex.pine@nyu.edu

set -x

module purge

# this works on crunchy
#module load LuaJIT-2.0.4

# TODO hpc only?
module load torch-deps/7
module load torch/intel/20151009

cd ~/deeplearning/hw4/
th result.lua --mode=train --max_epoch=5 --model_type=gru
