# !/bin/sh
source "/work1/s144077/train_env/bin/activate"
###  Name of jobscript 
#BSUB -J testv8
### -- GPU --
#BSUB -q gpuv100
### -- How many gpus and hvor to run --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- CPU cores --
#BSUB -n 1
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:30
### -- How many nodes the job requests -- ##
#BSUB -R "span[hosts=1]" 
### -- How much ram the requests -- ##
#BSUB -R "rusage[mem=1GB]" 
### -- How many nodes the job requests -- ##
# For requesting the extra big GPU w. 32GB of VRAM
#BSUB -R "select[gpu32gb]" 

#BSUB -N Send an email when done
#BSUB -o testv8/%J_log_fil.out Log file
#BSUB -e testv8/%J_Error_file.err Error log file

echo "Starting:"



echo "Starting:"
## Get directory
cd /work1/s144077/
. train_env/bin/activate train_env

CONTENTPATH=/testv8/
BATCH_SIZE=128
IMAGE_SIZE=64
echo $CONTENTPATH

nvidia-smi

#I run my python script with all of the settings
python3 deepflows/main.py --path $CONTENTPATH --image_size $IMAGE_SIZE --digit_size 28 --num_digits 1 --K 3 --L 2 --extractor_structure 32 conv 32 conv 64 --upscaler_structure 64 deconv-32 --x_dim $BATCH_SIZE 1 $IMAGE_SIZE $IMAGE_SIZE --condition_dim $BATCH_SIZE 1 $IMAGE_SIZE $IMAGE_SIZE