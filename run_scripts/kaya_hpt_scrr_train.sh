#!/bin/bash
#SBATCH --job-name=scoreinf_scrr_dual
#SBATCH --output=scoreinf_scrr_dual_progress_%A_%a.log
#SBATCH --error=scoreinf_scrr_dual_error_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-11
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au

module load Anaconda3/2024.06 gcc/11.5.0 cuda/12.6.3
module list
source activate bark_env

echo "Running on host: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "SLURM ID: $SLURM_ARRAY_ID $SLURM_ARRAY_TASK_ID"

# Make W&B monitor exactly the GPUs exposed to this job
export WANDB__SERVICE__GPU_MONITOR_POLICY=visible
export WANDB__SERVICE__GPU_MONITOR_DEVICES="$CUDA_VISIBLE_DEVICES"

# Torch Distributed defaults for single-node multi-GPU runs
export GPUS_PER_NODE=2
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-$((12000 + RANDOM % 20000))}

FOLDER_NAME=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
PROJECT_NAME=202510_hpt_smc
EXECUTABLE=$HOME/${PROJECT_NAME}
SCRATCH=$MYSCRATCH/${PROJECT_NAME}/$FOLDER_NAME
RESULTS=$MYGROUP/${PROJECT_NAME}_results/$FOLDER_NAME

mkdir -p "$SCRATCH" "$RESULTS"
echo "SCRATCH is $SCRATCH"
echo "RESULTS dir is $RESULTS"

echo "Copy path $EXECUTABLE to $SCRATCH"
cp -r "$EXECUTABLE" "$SCRATCH"
cd "$SCRATCH/$PROJECT_NAME" || exit 1

WORKSPACE_DIR="$SCRATCH/$PROJECT_NAME/workspaces"
mkdir -p "$WORKSPACE_DIR"

DATA_SRC="$MYSCRATCH/202510_hpt_data/workspaces/hdf5s"
DATA_VIEW="$WORKSPACE_DIR/hdf5s"
ln -s "$DATA_SRC" "$DATA_VIEW"

#############################################
# SCRR-only dual-GPU jobs:
# total = 3 adapters x 1 losses x 4 input settings = 12.
ADAPTERS=("hpt" "hppnet" "dynest")
LOSSES=("kim_bce_l1")
COND_INPUT2=("onset" "frame" "onset" "onset")
COND_INPUT3=("null"  "null"  "frame" "exframe")
METHOD="scrr"

EXP_ADAPTER=()
EXP_LOSS=()
EXP_INPUT2=()
EXP_INPUT3=()

for ADAPTER in "${ADAPTERS[@]}"; do
  for LOSS_TYPE in "${LOSSES[@]}"; do
    for i in "${!COND_INPUT2[@]}"; do
      EXP_ADAPTER+=("$ADAPTER")
      EXP_LOSS+=("$LOSS_TYPE")
      EXP_INPUT2+=("${COND_INPUT2[$i]}")
      EXP_INPUT3+=("${COND_INPUT3[$i]}")
    done
  done
done

TOTAL_JOBS=${#EXP_ADAPTER[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (0..$((TOTAL_JOBS-1)))"
  exit 1
fi

ADAPTER=${EXP_ADAPTER[$SLURM_ARRAY_TASK_ID]}
LOSS_TYPE=${EXP_LOSS[$SLURM_ARRAY_TASK_ID]}
INPUT2=${EXP_INPUT2[$SLURM_ARRAY_TASK_ID]}
INPUT3=${EXP_INPUT3[$SLURM_ARRAY_TASK_ID]}

echo "Adapter: $ADAPTER"
echo "Method : $METHOD"
echo "Loss   : $LOSS_TYPE"
echo "Input2 : $INPUT2"
echo "Input3 : $INPUT3"

LAUNCHER="torchrun --standalone --nnodes=1 --nproc_per_node=${GPUS_PER_NODE}"
CMD="$LAUNCHER pytorch/train_score_inf_dual.py \
  exp.workspace=\"$WORKSPACE_DIR\" \
  +exp.use_fsdp=true \
  model.type=\"$ADAPTER\" \
  score_informed.method=\"$METHOD\" \
  model.input2=\"$INPUT2\" \
  model.input3=\"$INPUT3\" \
  loss.loss_type=\"$LOSS_TYPE\""

echo "Running: $CMD"
eval "$CMD"

#############################################
[ -d "$WORKSPACE_DIR/checkpoints" ] && mv "$WORKSPACE_DIR/checkpoints/" "${RESULTS}/"
[ -d "$WORKSPACE_DIR/logs" ] && mv "$WORKSPACE_DIR/logs/" "${RESULTS}/"

cd "$HOME" || exit 1
rm -r "$SCRATCH"
conda deactivate 2>/dev/null || deactivate 2>/dev/null || true
echo scoreinf_scrr_dual $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished at `date`
