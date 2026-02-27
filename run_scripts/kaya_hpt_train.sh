#!/bin/bash
#SBATCH --job-name=scoreinf_ablate
#SBATCH --output=scoreinf_progress_%A_%a.log
#SBATCH --error=scoreinf_error_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-26
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au

module load Anaconda3/2024.06 cuda/11.8 gcc/11.5.0
module list
source activate bark_env #hpt_mamba

echo "Running on host: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "SLURM ID: $SLURM_ARRAY_ID $SLURM_ARRAY_TASK_ID"

FOLDER_NAME=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
PROJECT_NAME=202510_hpt_smc
EXECUTABLE=$HOME/${PROJECT_NAME}
SCRATCH=$MYSCRATCH/${PROJECT_NAME}/$FOLDER_NAME
RESULTS=$MYGROUP/${PROJECT_NAME}_results/$FOLDER_NAME

mkdir -p $SCRATCH $RESULTS
echo SCRATCH is $SCRATCH
echo RESULTS dir is $RESULTS

echo "Copy path $EXECUTABLE to $SCRATCH"
cp -r $EXECUTABLE $SCRATCH
cd $SCRATCH/$PROJECT_NAME

WORKSPACE_DIR=$SCRATCH/$PROJECT_NAME/workspaces
mkdir -p $WORKSPACE_DIR

DATA_SRC=$MYSCRATCH/202510_hpt_data/workspaces/hdf5s
DATA_VIEW=$WORKSPACE_DIR/hdf5s
ln -s $DATA_SRC $DATA_VIEW

#############################################
# Adapter/method/loss/input ablation (scrr + note_editor moved to separate scripts):
# direct: input2=null, input3=null
# dual_gated/bilstm: (onset,null), (frame,null), (onset,frame), (onset,exframe)
# Total = 3 adapters x (1*1*1 + 2*1*4) = 27 array jobs.
ADAPTERS=("hpt" "hppnet" "dynest")
METHODS=("direct" "dual_gated" "bilstm")
LOSSES=("kim_bce_l1")
COND_INPUT2=("onset" "frame" "onset" "onset")
COND_INPUT3=("null"  "null"  "frame" "exframe")

EXP_ADAPTER=()
EXP_METHOD=()
EXP_LOSS=()
EXP_INPUT2=()
EXP_INPUT3=()

for ADAPTER in "${ADAPTERS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    for LOSS_TYPE in "${LOSSES[@]}"; do
      if [ "$METHOD" = "direct" ]; then
        EXP_ADAPTER+=("$ADAPTER")
        EXP_METHOD+=("$METHOD")
        EXP_LOSS+=("$LOSS_TYPE")
        EXP_INPUT2+=("null")
        EXP_INPUT3+=("null")
      else
        for i in "${!COND_INPUT2[@]}"; do
          EXP_ADAPTER+=("$ADAPTER")
          EXP_METHOD+=("$METHOD")
          EXP_LOSS+=("$LOSS_TYPE")
          EXP_INPUT2+=("${COND_INPUT2[$i]}")
          EXP_INPUT3+=("${COND_INPUT3[$i]}")
        done
      fi
    done
  done
done

TOTAL_JOBS=${#EXP_ADAPTER[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (0..$((TOTAL_JOBS-1)))"
  exit 1
fi

ADAPTER=${EXP_ADAPTER[$SLURM_ARRAY_TASK_ID]}
METHOD=${EXP_METHOD[$SLURM_ARRAY_TASK_ID]}
LOSS_TYPE=${EXP_LOSS[$SLURM_ARRAY_TASK_ID]}
INPUT2=${EXP_INPUT2[$SLURM_ARRAY_TASK_ID]}
INPUT3=${EXP_INPUT3[$SLURM_ARRAY_TASK_ID]}

echo "Adapter: $ADAPTER"
echo "Method : $METHOD"
echo "Loss   : $LOSS_TYPE"
echo "Input2 : $INPUT2"
echo "Input3 : $INPUT3"

python pytorch/train_score_inf.py \
  exp.workspace="$WORKSPACE_DIR" \
  model.input2="$INPUT2" \
  model.input3="$INPUT3" \
  model.type="$ADAPTER" \
  score_informed.method="$METHOD" \
  loss.loss_type="$LOSS_TYPE"

#############################################
mv "$WORKSPACE_DIR/checkpoints/" "${RESULTS}/"
mv "$WORKSPACE_DIR/logs/" "${RESULTS}/"
cd $HOME
rm -r $SCRATCH # clean up the scratch space
source deactivate # Deactivate the conda environment - source or conda deactivate
echo scoreinf_ablate $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished at `date`
