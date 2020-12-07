# !/bin/sh
###  Name of jobscript 
#BSUB -J name
### -- GPU --
#BSUB -q gpuv100
### -- How many gpus and how to run --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- CPU cores --
#BSUB -n 1
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
### -- How many nodes the job requests -- ##
#BSUB -R "span[hosts=1]" 
### -- How much ram the requests -- ##
#BSUB -R "rusage[mem=32GB]" 
### -- How many nodes the job requests -- ##
# For requesting the extra big GPU w. 32GB of VRAM
#BSUB -R "select[gpu32gb]" 

#BSUB -N Send an email when done
#BSUB -o thesedoesntwork/%J_log_fil.out Log file
#BSUB -e thesedoesntwork/%J_Error_file.err Error log file



echo "Starting:"
## Get directory
cd /work1/s144077/
. train_env/bin/activate train_env

CONTENTPATH=/content/
LR=0.0001
PATIENCE_LR=20

NBITS=6

BETA_MAX=1
BETA_MIN=0.0001
BETA_STEPS=50000

HDIM=200
ZDIM=56

DATASET=mnist

BATCH_SIZE=128
NUM_WORKERS=2
IMAGE_SIZE=64

DIGIT_SIZE=28
NUM_DIGITS=2
STEPLENGTH=4


CHANNELS=1
K_SIZE=10
L_SIZE=3
N_UNITS_PRIOR=512
AFFINEHIDDEN=128
#'actnorm','batchnorm'
FLOWNORM=actnorm
FRAMES=6

#"instancenorm", "batchnorm", "none"
DOWNUPSCALERNORM=batchnorm


#"instancenorm", "batchnorm", "none"
NORM_TYPE=none
TEMPERATURE=0.7
SCALER=2





EPOCHS=100000000


echo $CONTENTPATH

nvidia-smi

#I run my python script with all of the settings
python3 deepflows/main.py --extractor_structure 1-2-squeeze 8-8-squeeze 32-32-squeeze --upscaler_structure 64 squeeze-32-32 squeeze-16-16 --prior_structure 256 128 --encoder_structure 256 128 --make_conditional --learn_prior --skip_connection --flow_norm $FLOWNORM --step_length $STEPLENGTH --structure_scaler $SCALER --choose_data $DATASET --n_units_affine $AFFINEHIDDEN --n_units_prior $N_UNITS_PRIOR --temperature $TEMPERATURE --norm_type $NORM_TYPE --z_dim $ZDIM --h_dim $HDIM --beta_min $BETA_MIN --beta_steps $BETA_STEPS --beta_max $BETA_MAX --learning_rate $LR --n_epochs $EPOCHS --n_bits $NBITS --n_frames $FRAMES --path $CONTENTPATH --image_size $IMAGE_SIZE --digit_size $DIGIT_SIZE --num_digits $NUM_DIGITS --K $K_SIZE --L $L_SIZE --x_dim $BATCH_SIZE $CHANNELS $IMAGE_SIZE $IMAGE_SIZE --condition_dim $BATCH_SIZE $CHANNELS $IMAGE_SIZE $IMAGE_SIZE --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --patience_lr $PATIENCE_LR --norm_type_coders $DOWNUPSCALERNORM
