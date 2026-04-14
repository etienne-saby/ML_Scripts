#!/bin/bash
# ==============================================================================
# FILE    : run_extraction.sh
# PURPOSE : SLURM batch launcher for one stage of the Hi-sAFe Meta-Model
#           extraction pipeline.  Called by submit_extraction.sh; not intended
#           to be submitted directly in the new four-stage pipeline.
#
#           Maps SLURM_ARRAY_TASK_ID to an extraction mode:
#             task 0  →  AF
#             task 1  →  TA
#             task 2  →  TF
#             unset   →  ALL   (single-node fallback)
#
# AUTHOR  : Étienne SABY
# DATE    : 2026
#
# ------------------------------------------------------------------------------
# USAGE
# ------------------------------------------------------------------------------
#
# Single-node, all modes sequentially (small campaigns or testing):
#   sbatch --cpus-per-task=64 run_extraction.sh <campaign_name>
#
# Array mode — one node per extraction mode (called by submit_extraction.sh):
#   sbatch --array=0-2 --cpus-per-task=64 run_extraction.sh <campaign_name>
#
# ------------------------------------------------------------------------------
# RECOMMENDED RESOURCE GUIDELINES
# ------------------------------------------------------------------------------
#   < 500 simulations  :  --cpus-per-task=32   --mem=64G    --time=01:00:00
#   500 – 2 000 sims   :  --cpus-per-task=64   --mem=128G   --time=04:00:00
#   > 2 000 sims       :  --cpus-per-task=96   --mem=256G   --time=08:00:00
# ==============================================================================

#SBATCH --job-name=hisafe_ext
#SBATCH --account=ondemand@hisafe
#SBATCH --qos=cpu-ondemand-long
#SBATCH --output=logs_extraction/logs_ext_%A_%a.out
#SBATCH --error=logs_extraction/logs_ext_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
# --cpus-per-task must be set at submission time: sbatch --cpus-per-task=N ...

set -euo pipefail

# ------------------------------------------------------------------------------
# 0. ARGUMENT VALIDATION
# ------------------------------------------------------------------------------
CAMPAIGN_NAME=${1:?'ERROR: campaign_name is required.
Usage: sbatch --cpus-per-task=N [--array=0-2] run_extraction.sh <campaign_name>'}

# ------------------------------------------------------------------------------
# 1. CONDA INITIALIZATION
# ------------------------------------------------------------------------------
CONDA_HOME="$HOME/miniconda3"
if [[ ! -d "$CONDA_HOME" ]]; then
  echo "ERROR: Conda not found at $CONDA_HOME"
  exit 1
fi

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate r_hisafe || { echo "ERROR: Failed to activate 'r_hisafe'"; exit 1; }

mkdir -p "logs_extraction"

# ------------------------------------------------------------------------------
# 2. ARRAY TASK → MODE MAPPING
#    SLURM_ARRAY_TASK_ID maps to an extraction mode:
#      0 = AF  |  1 = TA  |  2 = TF  |  unset = ALL
# ------------------------------------------------------------------------------
ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-"all"}
MODE_FLAG=""

if [[ "${ARRAY_TASK_ID}" != "all" ]]; then
  case "${ARRAY_TASK_ID}" in
    0) MODE_FLAG="--mode AF" ;;
    1) MODE_FLAG="--mode TA" ;;
    2) MODE_FLAG="--mode TF" ;;
    *)
      echo "ERROR: SLURM_ARRAY_TASK_ID=${ARRAY_TASK_ID} is invalid (expected 0, 1, or 2)."
      exit 1
      ;;
  esac
  echo "Array task ${ARRAY_TASK_ID} → mode ${MODE_FLAG#--mode }"
fi

# ------------------------------------------------------------------------------
# 3. RESOURCE AUTO-DETECTION
# ------------------------------------------------------------------------------
N_CPUS=${SLURM_CPUS_PER_TASK:-1}
MEM_MB=${SLURM_MEM_PER_NODE:-65536}   # SLURM provides value in MB

# Reserve ~10 % of RAM for OS and data-loading overhead.
MEM_R_MB=$(( MEM_MB * 90 / 100 ))

echo "============================================================"
echo "Hi-sAFe Meta-Model — Extraction"
echo "Campaign      : ${CAMPAIGN_NAME}"
echo "Node          : $(hostname)"
echo "Mode          : ${MODE_FLAG:-ALL}"
echo "CPUs allocated: ${N_CPUS}"
echo "RAM allocated : ${MEM_MB} MB  (available to R: ${MEM_R_MB} MB)"
echo "Start time    : $(date)"
echo "============================================================"

# ------------------------------------------------------------------------------
# 4. ADAPTIVE CHUNK SIZE
#    Heuristic: ~4 MB per simulation × 3 source files × chunk.
#    Targets chunks that fit comfortably in RAM during rbindlist binding.
#    Formula: chunk_size = min(200, floor(MEM_R_MB / (4 × 3 × n_cores)))
#    Example: 128 GB RAM, 64 cores → min(200, floor(117964 / 768)) = 153
# ------------------------------------------------------------------------------
CHUNK_SIZE=$(python3 -c "
mem  = ${MEM_R_MB}
cpus = ${N_CPUS}
raw  = mem // (20 * 3 * max(1, cpus))
print(min(200, max(25, raw)))
" 2>/dev/null || echo 75)

echo "Chunk size (adaptive): ${CHUNK_SIZE}"
echo ""

# ------------------------------------------------------------------------------
# 5. LAUNCH R SCRIPT
# ------------------------------------------------------------------------------
SCRIPT_DIR=/scratch/projects/hisafe/MetAIsAFe/01_Scripts/R/02_DataExtraction

cd "${SCRIPT_DIR}" \
  || { echo "ERROR: script directory not found: ${SCRIPT_DIR}"; exit 1; }

Rscript --vanilla 03_cluster_extract.R \
  "${CAMPAIGN_NAME}"       \
  --cpus  "${N_CPUS}"      \
  --chunk "${CHUNK_SIZE}"  \
  ${MODE_FLAG}

EXIT_CODE=$?

# ------------------------------------------------------------------------------
# 6. EXIT REPORT
# ------------------------------------------------------------------------------
echo ""
echo "============================================================"
if [[ ${EXIT_CODE} -ne 0 ]]; then
  echo "ERROR: job failed with exit code ${EXIT_CODE}  ($(date))"
  exit ${EXIT_CODE}
fi
echo "Job completed successfully  ($(date))"
echo "============================================================"