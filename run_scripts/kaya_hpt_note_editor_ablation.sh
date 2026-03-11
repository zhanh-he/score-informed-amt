#!/bin/bash
#SBATCH --job-name=scoreinf_note_arch_ablation
#SBATCH --output=scoreinf_note_arch_ablation_progress_%A_%a.log
#SBATCH --error=scoreinf_note_arch_ablation_error_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-19
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au

module load Anaconda3/2024.06 gcc/11.5.0 cuda/12.4.1 # cuda/12.6.3
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
# note_editor architecture-focused ablation
# fixed: adapter=hpt, input2=onset, input3=frame, loss=kim_bce_l1
# total: 20 jobs (0..19)
#
# A0-A9   : transformer vs conformer across depth/width
# B0-B3   : head-count sensitivity
# C0-C3   : dropout + alpha sensitivity
# D0-D1   : train schedule sensitivity
#############################################

ADAPTER="hpt"
METHOD="note_editor"
LOSS_TYPE="kim_bce_l1"
INPUT2_FIXED="onset"
INPUT3_FIXED="frame"
SEED_FIXED="86"

EXP_NAME=()
EXP_ARCH=()
EXP_DMODEL=()
EXP_NLAYERS=()
EXP_NHEADS=()
EXP_DROPOUT=()
EXP_ALPHA=()
EXP_TRAIN_MODE=()
EXP_SWITCH_IT=()
EXP_SEED=()

add_exp() {
  EXP_NAME+=("$1")
  EXP_ARCH+=("$2")
  EXP_DMODEL+=("$3")
  EXP_NLAYERS+=("$4")
  EXP_NHEADS+=("$5")
  EXP_DROPOUT+=("$6")
  EXP_ALPHA+=("$7")
  EXP_TRAIN_MODE+=("$8")
  EXP_SWITCH_IT+=("$9")
  EXP_SEED+=("${10}")
}

# Group A: architecture x capacity
add_exp "A0_conf_d128_l2_h4" "conformer"   "128" "2" "4" "0.1" "0.2" "joint" "100000" "$SEED_FIXED"
add_exp "A1_trf_d128_l2_h4"  "transformer" "128" "2" "4" "0.1" "0.2" "joint" "100000" "$SEED_FIXED"
add_exp "A2_conf_d128_l1_h4" "conformer"   "128" "1" "4" "0.1" "0.2" "joint" "100000" "$SEED_FIXED"
add_exp "A3_trf_d128_l1_h4"  "transformer" "128" "1" "4" "0.1" "0.2" "joint" "100000" "$SEED_FIXED"
add_exp "A4_conf_d128_l4_h4" "conformer"   "128" "4" "4" "0.1" "0.2" "joint" "100000" "$SEED_FIXED"
add_exp "A5_trf_d128_l4_h4"  "transformer" "128" "4" "4" "0.1" "0.2" "joint" "100000" "$SEED_FIXED"
add_exp "A6_conf_d64_l2_h4"  "conformer"   "64"  "2" "4" "0.1" "0.2" "joint" "100000" "$SEED_FIXED"
add_exp "A7_trf_d64_l2_h4"   "transformer" "64"  "2" "4" "0.1" "0.2" "joint" "100000" "$SEED_FIXED"
add_exp "A8_conf_d256_l2_h8" "conformer"   "256" "2" "8" "0.1" "0.2" "joint" "100000" "$SEED_FIXED"
add_exp "A9_trf_d256_l2_h8"  "transformer" "256" "2" "8" "0.1" "0.2" "joint" "100000" "$SEED_FIXED"

# Group B: head count sensitivity
add_exp "B0_conf_d128_l2_h2" "conformer"   "128" "2" "2" "0.1" "0.2" "joint" "100000" "$SEED_FIXED"
add_exp "B1_conf_d128_l2_h8" "conformer"   "128" "2" "8" "0.1" "0.2" "joint" "100000" "$SEED_FIXED"
add_exp "B2_trf_d128_l2_h2"  "transformer" "128" "2" "2" "0.1" "0.2" "joint" "100000" "$SEED_FIXED"
add_exp "B3_trf_d128_l2_h8"  "transformer" "128" "2" "8" "0.1" "0.2" "joint" "100000" "$SEED_FIXED"

# Group C: dropout + alpha sensitivity
add_exp "C0_conf_dropout0"   "conformer" "128" "2" "4" "0.0" "0.2" "joint" "100000" "$SEED_FIXED"
add_exp "C1_conf_dropout03"  "conformer" "128" "2" "4" "0.3" "0.2" "joint" "100000" "$SEED_FIXED"
add_exp "C2_conf_alpha01"    "conformer" "128" "2" "4" "0.1" "0.1" "joint" "100000" "$SEED_FIXED"
add_exp "C3_conf_alpha04"    "conformer" "128" "2" "4" "0.1" "0.4" "joint" "100000" "$SEED_FIXED"

# Group D: schedule sensitivity
add_exp "D0_conf_adapter_then_score_sw30k" "conformer" "128" "2" "4" "0.1" "0.2" "adapter_then_score" "30000" "$SEED_FIXED"
add_exp "D1_conf_adapter_then_joint_sw30k" "conformer" "128" "2" "4" "0.1" "0.2" "adapter_then_joint" "30000" "$SEED_FIXED"

TOTAL_JOBS=${#EXP_NAME[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (0..$((TOTAL_JOBS-1)))"
  exit 1
fi

EXP_TAG=${EXP_NAME[$SLURM_ARRAY_TASK_ID]}
ARCH=${EXP_ARCH[$SLURM_ARRAY_TASK_ID]}
DMODEL=${EXP_DMODEL[$SLURM_ARRAY_TASK_ID]}
NLAYERS=${EXP_NLAYERS[$SLURM_ARRAY_TASK_ID]}
NHEADS=${EXP_NHEADS[$SLURM_ARRAY_TASK_ID]}
DROPOUT=${EXP_DROPOUT[$SLURM_ARRAY_TASK_ID]}
ALPHA=${EXP_ALPHA[$SLURM_ARRAY_TASK_ID]}
TRAIN_MODE=${EXP_TRAIN_MODE[$SLURM_ARRAY_TASK_ID]}
SWITCH_IT=${EXP_SWITCH_IT[$SLURM_ARRAY_TASK_ID]}
SEED=${EXP_SEED[$SLURM_ARRAY_TASK_ID]}

echo "ExpTag : $EXP_TAG"
echo "Adapter: $ADAPTER"
echo "Method : $METHOD"
echo "Loss   : $LOSS_TYPE"
echo "Input2 : $INPUT2_FIXED"
echo "Input3 : $INPUT3_FIXED"
echo "arch   : $ARCH"
echo "d_model: $DMODEL"
echo "layers : $NLAYERS"
echo "heads  : $NHEADS"
echo "dropout: $DROPOUT"
echo "alpha  : $ALPHA"
echo "tmode  : $TRAIN_MODE"
echo "switch : $SWITCH_IT"
echo "seed   : $SEED"

python pytorch/train_score_inf.py \
  exp.workspace="$WORKSPACE_DIR" \
  exp.random_seed="$SEED" \
  model.input2="$INPUT2_FIXED" \
  model.input3="$INPUT3_FIXED" \
  model.type="$ADAPTER" \
  score_informed.method="$METHOD" \
  ++score_informed.train_mode="$TRAIN_MODE" \
  ++score_informed.switch_iteration="$SWITCH_IT" \
  ++score_informed.params.arch="$ARCH" \
  ++score_informed.params.d_model="$DMODEL" \
  ++score_informed.params.n_layers="$NLAYERS" \
  ++score_informed.params.n_heads="$NHEADS" \
  ++score_informed.params.dropout="$DROPOUT" \
  ++score_informed.params.alpha="$ALPHA" \
  loss.loss_type="$LOSS_TYPE" \
  wandb.comment="$EXP_TAG"

#############################################
[ -d "$WORKSPACE_DIR/checkpoints" ] && mv "$WORKSPACE_DIR/checkpoints/" "${RESULTS}/"
[ -d "$WORKSPACE_DIR/logs" ] && mv "$WORKSPACE_DIR/logs/" "${RESULTS}/"
cd $HOME
rm -r $SCRATCH
conda deactivate 2>/dev/null || source deactivate 2>/dev/null || deactivate 2>/dev/null || true
echo scoreinf_note_arch_ablation $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished at `date`