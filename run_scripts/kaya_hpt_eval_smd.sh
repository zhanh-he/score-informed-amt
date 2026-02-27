#!/bin/bash
#SBATCH --job-name=scoreinf_eval
#SBATCH --output=scoreinf_eval_%A_%a.log
#SBATCH --error=scoreinf_eval_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --array=0-35
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au

# Usage:
#   sbatch run_scripts/kaya_hpt_eval_smd.sh
#   sbatch --array=0-35 run_scripts/kaya_hpt_eval_smd.sh
#   bash run_scripts/kaya_hpt_eval_smd.sh  # local sequential run
# Args:
#   1: dataset.test_set      (default: smd)
#   2: end iteration         (default: 120000)
#   3: step iteration        (default: 10000)


module load Anaconda3/2024.06 gcc/11.5.0 cuda/12.4.1
module list
source activate bark_env

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
rm -rf "$DATA_VIEW"
ln -s $DATA_SRC $DATA_VIEW

CKPT_SRC=$MYSCRATCH/202510_hpt_data/workspaces/checkpoints
CKPT_VIEW=$WORKSPACE_DIR/checkpoints
rm -rf "$CKPT_VIEW"
ln -s $CKPT_SRC $CKPT_VIEW

echo "HDF5 link -> $(readlink -f "$DATA_VIEW")"
echo "CKPT link -> $(readlink -f "$CKPT_VIEW")"
echo "CKPT hpt/0 exists? -> $(ls "$CKPT_VIEW/hpt/0_iterations.pth" 2>/dev/null || echo NO)"

TEST_SET=${1:-smd}
END_ITER=${2:-120000}
STEP_ITER=${3:-10000}

# ----------------------------
# Eval matrix
# ----------------------------
ADAPTERS=("hpt" "hppnet" "dynest")
METHODS=(
  "direct"
  "note_editor"
  "bilstm"
  "dual_gated"
  # "scrr"   # reserved: currently not trained well
)

# Candidate input pairs:
# (null,null), (onset,null), (frame,null), (onset,frame), (onset,exframe)
PAIR_INPUT2=("null" "onset" "frame" "onset" "onset")
PAIR_INPUT3=("null" "null"  "null"  "frame" "exframe")

EXP_ADAPTER=()
EXP_METHOD=()
EXP_INPUT2=()
EXP_INPUT3=()

for ADAPTER in "${ADAPTERS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    if [ "$METHOD" = "direct" ]; then
      # direct only supports (null, null)
      EXP_ADAPTER+=("$ADAPTER")
      EXP_METHOD+=("$METHOD")
      EXP_INPUT2+=("${PAIR_INPUT2[0]}")
      EXP_INPUT3+=("${PAIR_INPUT3[0]}")
    elif [ "$METHOD" = "note_editor" ]; then
      # note_editor supports: (onset,null), (onset,frame), (onset,exframe)
      for IDX in 1 3 4; do
        EXP_ADAPTER+=("$ADAPTER")
        EXP_METHOD+=("$METHOD")
        EXP_INPUT2+=("${PAIR_INPUT2[$IDX]}")
        EXP_INPUT3+=("${PAIR_INPUT3[$IDX]}")
      done
    else
      # bilstm / dual_gated / (future scrr):
      # (onset,null), (frame,null), (onset,frame), (onset,exframe)
      for IDX in 1 2 3 4; do
        EXP_ADAPTER+=("$ADAPTER")
        EXP_METHOD+=("$METHOD")
        EXP_INPUT2+=("${PAIR_INPUT2[$IDX]}")
        EXP_INPUT3+=("${PAIR_INPUT3[$IDX]}")
      done
    fi
  done
done

TOTAL_JOBS=${#EXP_ADAPTER[@]}

run_one() {
  local IDX=$1
  local MODEL_TYPE=${EXP_ADAPTER[$IDX]}
  local SCORE_METHOD=${EXP_METHOD[$IDX]}
  local INPUT2=${EXP_INPUT2[$IDX]}
  local INPUT3=${EXP_INPUT3[$IDX]}

  echo "Host            : $(hostname)"
  echo "CUDA            : $CUDA_VISIBLE_DEVICES"
  echo "Project root    : $SCRATCH/$PROJECT_NAME"
  echo "Workspace       : $WORKSPACE_DIR"
  echo "Eval dataset    : $TEST_SET"
  echo "Iteration sweep : 0 -> $END_ITER (step=$STEP_ITER)"
  echo "Job idx         : $IDX / $((TOTAL_JOBS-1))"
  echo "Model config    : type=$MODEL_TYPE input2=$INPUT2 input3=$INPUT3 method=$SCORE_METHOD"

  python pytorch/calculate_scores.py \
    exp.workspace="$WORKSPACE_DIR" \
    dataset.test_set=$TEST_SET \
    model.type=$MODEL_TYPE \
    model.input2=$INPUT2 \
    model.input3=$INPUT3 \
    score_informed.method=$SCORE_METHOD \
    exp.run_infer=multi \
    +exp.eval_start_iteration=0 \
    +exp.eval_end_iteration=$END_ITER \
    +exp.eval_step_iteration=$STEP_ITER
}

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (0..$((TOTAL_JOBS-1)))"
  exit 1
fi

run_one "$SLURM_ARRAY_TASK_ID"

#############################################
[ -d "$WORKSPACE_DIR/kim_eval" ] && mv "$WORKSPACE_DIR/kim_eval/" "${RESULTS}/"
[ -d "$WORKSPACE_DIR/kim_eval_summary" ] && mv "$WORKSPACE_DIR/kim_eval_summary/" "${RESULTS}/"
[ -d "$WORKSPACE_DIR/logs" ] && mv "$WORKSPACE_DIR/logs/" "${RESULTS}/"
cd $HOME
rm -r $SCRATCH
conda deactivate
echo scoreinf_eval $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished at `date`
