#!/bin/bash
# ==============================================================================
# FILE    : submit_extraction.sh
# PURPOSE : Master submission script for the Hi-sAFe Meta-Model extraction
#           pipeline.  Chains four SLURM stages with dependency management:
#
#             Stage 1  PHENO       — phenological calendar (plan_df only, ~secs)
#             Stage 2  AF + TA + TF — raw extraction + cycle aggregation,
#                                     all three modes run in parallel
#             Stage 3  BUILD       — meta-model table assembly
#
#           Dependency chain:
#             PHENO  →  AF, TA, TF  (all three start once PHENO completes)
#             AF + TA + TF  →  BUILD
#
#           Any stage failure automatically cancels all downstream jobs.
#
# USAGE
# ------------------------------------------------------------------------------
#   bash submit_extraction.sh <campaign_name> [options]
#
# OPTIONS
#   --cpus  N          CPUs per extraction node (default: 64)
#   --mem   XG         Memory per extraction node (default: 128G)
#   --time  HH:MM:SS   Wall time for extraction jobs (default: 08:00:00)
#   --dry-run          Print sbatch commands without submitting
#
# EXAMPLES
#   bash submit_extraction.sh sobol_S11111_n2048
#   bash submit_extraction.sh sobol_S11111_n2048 --cpus 96 --mem 256G
#   bash submit_extraction.sh sobol_S11111_n2048 --dry-run
#
# NOTES
#   • run_extraction.sh must be in the same directory as this script.
#   • PHENO and BUILD stages use fixed lightweight resources (8 CPUs / 16 G)
#     regardless of the --cpus / --mem options, which apply to extraction only.
#   • All job IDs are recorded in logs_extraction/submit_<campaign>_<date>.log.
#     Cancel the entire pipeline with: scancel <JOB_PHENO> <JOB_EXT> <JOB_BUILD>
#
# AUTHOR  : Étienne SABY
# DATE    : 2026
# ==============================================================================

set -euo pipefail

# ==============================================================================
# 0. ARGUMENT PARSING
# ==============================================================================

usage() {
  echo "Usage: bash submit_extraction.sh <campaign_name> [--cpus N] [--mem XG] [--time HH:MM:SS] [--dry-run]"
  exit 1
}

CAMPAIGN_NAME=${1:-""}
if [[ -z "${CAMPAIGN_NAME}" || "${CAMPAIGN_NAME}" == --* ]]; then
  echo "ERROR: campaign_name is required as the first positional argument."
  usage
fi
shift

CPUS="64"
MEM="128G"
WALLTIME="08:00:00"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpus)    CPUS="$2";     shift 2 ;;
    --mem)     MEM="$2";      shift 2 ;;
    --time)    WALLTIME="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true;  shift   ;;
    *)
      echo "ERROR: Unknown option: $1"
      usage
      ;;
  esac
done

# ==============================================================================
# 1. ENVIRONMENT CHECKS
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/run_extraction.sh"

if [[ ! -f "${LAUNCHER}" ]]; then
  echo "ERROR: run_extraction.sh not found at: ${LAUNCHER}"
  echo "       Both scripts must be in the same directory."
  exit 1
fi

command -v sbatch &>/dev/null \
  || { echo "ERROR: sbatch not found — are you on a SLURM login node?"; exit 1; }

mkdir -p "logs_extraction"

# ==============================================================================
# 2. SUBMISSION LOG SETUP
# ==============================================================================

SUBMIT_DATE=$(date +"%Y%m%d_%H%M%S")
SUBMIT_LOG="logs_extraction/submit_${CAMPAIGN_NAME}_${SUBMIT_DATE}.log"

log() {
  # Write to stderr AND the submission log.
  # stderr ensures log() output is never captured by $() in submit_job().
  echo "$*" | tee -a "${SUBMIT_LOG}" >&2
}

log "============================================================"
log "Hi-sAFe Meta-Model — Pipeline Submission"
log "Campaign        : ${CAMPAIGN_NAME}"
log "Extraction CPUs : ${CPUS}"
log "Extraction RAM  : ${MEM}"
log "Wall time       : ${WALLTIME}"
log "Dry run         : ${DRY_RUN}"
log "Launcher        : ${LAUNCHER}"
log "Submitted       : $(date)"
log "============================================================"
log ""

# ==============================================================================
# 3. SBATCH HELPER
#    submit_job <label> <extra_sbatch_args> <script_args>
#    Prints human-readable command in dry-run mode.
#    Returns only the job ID on stdout (safe for $() capture).
# ==============================================================================

submit_job() {
  local label="$1"
  local extra_sbatch="$2"   # e.g. "--array=0-2 --dependency=afterok:123"
  local script_args="$3"    # passed after the launcher path

  # Build argument list as an array: avoids backslash-newline wrapping in logs.
  local -a args=(
    sbatch
    --job-name="extract_${label}"
    --account=ondemand@hisafe
    --qos=cpu-ondemand-long
    --ntasks=1
    --cpus-per-task="${CPUS}"
    --mem="${MEM}"
    --time="${WALLTIME}"
    --output="logs_extraction/logs_${label}_%A_%a.out"
    --error="logs_extraction/logs_${label}_%A_%a.out"
  )

  # Append optional extra SLURM flags (word-split intentional).
  # shellcheck disable=SC2206
  [[ -n "${extra_sbatch}" ]] && args+=( ${extra_sbatch} )

  # Append launcher and script arguments.
  args+=( "${LAUNCHER}" )
  [[ -n "${script_args}" ]] && args+=( "${script_args}" )

  if [[ "${DRY_RUN}" == true ]]; then
    log "[DRY-RUN] would run:"
    log "  ${args[*]}"
    echo "DRYRUN_${label}"
    return 0
  fi

  local output exit_code
  output=$( "${args[@]}" 2>&1 )
  exit_code=$?

  if [[ ${exit_code} -ne 0 ]]; then
    log "ERROR: sbatch failed for stage '${label}':"
    log "  ${output}"
    exit ${exit_code}
  fi

  local job_id
  job_id=$(echo "${output}" | awk '/Submitted batch job/{print $4}')

  if [[ -z "${job_id}" ]]; then
    log "ERROR: could not parse job ID from sbatch output: '${output}'"
    exit 1
  fi

  log "  [${label}] Submitted — job ID: ${job_id}"
  echo "${job_id}"
}

# Helper for lightweight jobs (PHENO and BUILD) that use --wrap instead of the
# launcher, with fixed reduced resources.
submit_wrap_job() {
  local label="$1"
  local dependency="$2"   # e.g. "afterok:123" or ""
  local wrap_cmd="$3"

  local wrap_cpus=8
  local wrap_mem=16G
  local wrap_time=00:30:00

  local dep_flag=""
  [[ -n "${dependency}" ]] && dep_flag="--dependency=${dependency}"

  declare -a args=(
    sbatch
    --job-name="hisafe_${label}"
    --account=ondemand@hisafe
    --qos=cpu-ondemand-long
    --ntasks=1
    --cpus-per-task="${wrap_cpus}"
    --mem="${wrap_mem}"
    --time="${wrap_time}"
    --output="logs_extraction/logs_${label}_%j.out"
    --error="logs_extraction/logs_${label}_%j.out"
  )

  [[ -n "${dep_flag}" ]] && args+=( "${dep_flag}" )
  args+=( --wrap="${wrap_cmd}" )

  if [[ "${DRY_RUN}" == true ]]; then
    log "[DRY-RUN] would run:"
    # Re-add --wrap with visible quotes for readability
    log "  ${args[*]::${#args[@]}-1} --wrap=\"${wrap_cmd}\""
    echo "DRYRUN_${label}"
    return 0
  fi

  local output exit_code
  output=$( "${args[@]}" 2>&1 )
  exit_code=$?

  if [[ ${exit_code} -ne 0 ]]; then
    log "ERROR: sbatch failed for stage '${label}':"
    log "  ${output}"
    exit ${exit_code}
  fi

  local job_id
  job_id=$(echo "${output}" | awk '/Submitted batch job/{print $4}')

  if [[ -z "${job_id}" ]]; then
    log "ERROR: could not parse job ID from sbatch output: '${output}'"
    exit 1
  fi

  log "  [${label}] Submitted — job ID: ${job_id}"
  echo "${job_id}"
}

# ==============================================================================
# 4. STAGE 1 — PHENO
#    Builds the phenological calendar from plan_df only.
#    Very fast (~seconds). No simulation files are read.
#    Lightweight resources: 8 CPUs, 16 GB, 30 min.
# ==============================================================================

log "--- Stage 1: Phenological calendar (PHENO) ---"

PHENO_CMD="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate r_hisafe && cd ${SCRIPT_DIR} && Rscript --vanilla 03_cluster_extract.R ${CAMPAIGN_NAME} --mode PHENO --cpus 8"

JOB_PHENO=$(submit_wrap_job \
  "pheno" \
  "" \
  "${PHENO_CMD}"
)

log "  PHENO job ID : ${JOB_PHENO}"
log ""

# ==============================================================================
# 5. STAGE 2 — AF + TA + TF  (parallel extraction, all depend on PHENO)
#    All three modes start simultaneously once PHENO has written the calendar.
#    Each runs on a dedicated node via SLURM array tasks 0–2.
# ==============================================================================

log "--- Stage 2: Extraction — AF + TA + TF in parallel (after PHENO) ---"

JOB_EXT=$(submit_job \
  "ext" \
  "--array=0-2 --dependency=afterok:${JOB_PHENO}" \
  "${CAMPAIGN_NAME}"
)

log "  Extraction job ID : ${JOB_EXT}  (tasks 0=AF, 1=TA, 2=TF)"
log ""

# ==============================================================================
# 6. STAGE 3 — BUILD
#    Assembles the meta-model table from the three cycle files.
#    Lightweight resources: 8 CPUs, 16 GB, 30 min.
#    Starts once ALL three extraction tasks (0, 1, 2) have completed.
# ==============================================================================

log "--- Stage 3: Meta-model table assembly (BUILD, after all extraction) ---"

BUILD_CMD="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate r_hisafe && cd ${SCRIPT_DIR} && Rscript --vanilla 03_cluster_extract.R ${CAMPAIGN_NAME} --mode BUILD --cpus 8"

JOB_BUILD=$(submit_wrap_job \
  "build" \
  "afterok:${JOB_EXT}" \
  "${BUILD_CMD}"
)

log "  BUILD job ID : ${JOB_BUILD}"
log ""

# ==============================================================================
# 7. SUMMARY
# ==============================================================================

log "============================================================"
log "Pipeline submitted successfully"
log ""
log "  Stage 1 — PHENO  : job ${JOB_PHENO}"
log "  Stage 2 — EXT    : job ${JOB_EXT}  (tasks 0=AF / 1=TA / 2=TF, after ${JOB_PHENO})"
log "  Stage 3 — BUILD  : job ${JOB_BUILD}  (after all of ${JOB_EXT})"
log ""
log "Monitor progress:"
log "  squeue -u \$USER"
log "  tail -f logs_extraction/logs_pheno_${JOB_PHENO}.out"
log "  tail -f logs_extraction/logs_ext_${JOB_EXT}_0.out   # AF"
log "  tail -f logs_extraction/logs_ext_${JOB_EXT}_1.out   # TA"
log "  tail -f logs_extraction/logs_ext_${JOB_EXT}_2.out   # TF"
log ""
log "Cancel entire pipeline:"
log "  scancel ${JOB_PHENO} ${JOB_EXT} ${JOB_BUILD}"
log ""
log "Full submission log: ${SUBMIT_LOG}"
log "============================================================"