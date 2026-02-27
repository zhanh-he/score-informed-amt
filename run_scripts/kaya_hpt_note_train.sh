#!/bin/bash
#SBATCH --job-name=scoreinf_note_editor
#SBATCH --output=scoreinf_note_editor_progress_%A_%a.log
#SBATCH --error=scoreinf_note_editor_error_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-8
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
# note_editor-only ablation:
# note_editor requires input2=onset; input3 in {null, frame, exframe}
# Total = 3 adapters x 1 losses x 3 input settings = 9 array jobs.
ADAPTERS=("hpt" "hppnet" "dynest")
LOSSES=("kim_bce_l1")
METHOD="note_editor"
INPUT2_FIXED="onset"
COND_INPUT3=("null" "frame" "exframe")

EXP_ADAPTER=()
EXP_LOSS=()
EXP_INPUT3=()

for ADAPTER in "${ADAPTERS[@]}"; do
  for LOSS_TYPE in "${LOSSES[@]}"; do
    for INPUT3 in "${COND_INPUT3[@]}"; do
      EXP_ADAPTER+=("$ADAPTER")
      EXP_LOSS+=("$LOSS_TYPE")
      EXP_INPUT3+=("$INPUT3")
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
INPUT3=${EXP_INPUT3[$SLURM_ARRAY_TASK_ID]}

echo "Adapter: $ADAPTER"
echo "Method : $METHOD"
echo "Loss   : $LOSS_TYPE"
echo "Input2 : $INPUT2_FIXED"
echo "Input3 : $INPUT3"

python pytorch/train_score_inf.py \
  exp.workspace="$WORKSPACE_DIR" \
  model.input2="$INPUT2_FIXED" \
  model.input3="$INPUT3" \
  model.type="$ADAPTER" \
  score_informed.method="$METHOD" \
  loss.loss_type="$LOSS_TYPE"

#############################################
mv "$WORKSPACE_DIR/checkpoints/" "${RESULTS}/"
mv "$WORKSPACE_DIR/logs/" "${RESULTS}/"
cd $HOME
rm -r $SCRATCH
source deactivate
echo scoreinf_note_editor $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished at `date`
