#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J classifier_rf_1
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 14:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=32GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the modules
module load ffmpeg/5.0.1

timeDir=time
mkdir -p $timeDir

#/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source deep_learning_env/bin/activate
/usr/bin/time -v -o $timeDir/classifier_rf_1.time python3 train_random_forest.py
