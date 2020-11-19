# !/bin/sh
###  Name of jobscript 
#BSUB -J testL5
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
#BSUB -o testL5/%J_log_fil.out Log file
#BSUB -e testL5/%J_Error_file.err Error log file



echo "Starting:"
## Get directory
cd /work1/s144077/
. train_env/bin/activate train_env

CONTENTPATH=/testL5/
LR=0.0001
NBITS=6

BETA_MAX=0.01
BETA_MIN=0.0001
BETA_STEPS=4000

HDIM=200
ZDIM=60

DATASET=mnist

BATCH_SIZE=128
IMAGE_SIZE=32

DIGIT_SIZE=21
NUM_DIGITS=2
STEPLENGTH=2

CHANNELS=1
K_SIZE=10
L_SIZE=5
FRAMES=6




#"instancenorm", "batchnorm", "none"
NORM_TYPE=none
TEMPERATURE=0.8
SCALER=2

AFFINEHIDDEN=128
N_UNITS_PRIOR=256



EPOCHS=100000000


# This is KINDA important #
EXTRACTED_FEATURES=256 

echo $CONTENTPATH

nvidia-smi

#I run my python script with all of the settings
python3 deepflows/main.py --extractor_structure 8 conv 16 conv 32 conv 64 conv 128 conv --upscaler_structure 256 deconv-128 deconv-64 deconv-32 deconv-16 --prior_structure 128 --encoder_structure 256 128 --make_conditional --learn_prior --step_length $STEPLENGTH --structure_scaler $SCALER --choose_data $DATASET --n_units_affine $AFFINEHIDDEN --n_units_prior $N_UNITS_PRIOR --temperature $TEMPERATURE --norm_type $NORM_TYPE --c_features $EXTRACTED_FEATURES --z_dim $ZDIM --h_dim $HDIM --beta_min $BETA_MIN --beta_steps $BETA_STEPS --beta_max $BETA_MAX --learning_rate $LR --n_epochs $EPOCHS --n_bits $NBITS --n_frames $FRAMES --path $CONTENTPATH --image_size $IMAGE_SIZE --digit_size $DIGIT_SIZE --num_digits $NUM_DIGITS --K $K_SIZE --L $L_SIZE --x_dim $BATCH_SIZE $CHANNELS $IMAGE_SIZE $IMAGE_SIZE --condition_dim $BATCH_SIZE $CHANNELS $IMAGE_SIZE $IMAGE_SIZE --batch_size $BATCH_SIZE
