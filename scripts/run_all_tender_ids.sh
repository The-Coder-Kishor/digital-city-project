#!/usr/bin/env bash

set -u -o pipefail

START_ID="${1:-000001}"
END_ID="${2:-999999}"

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_DIR}/downloads/batch_runs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
RATE_LIMIT_SECONDS="${RATE_LIMIT_SECONDS:-2.0}"
EXTRA_ARGS=("$@")

detect_max_parallel() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  if command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
    return
  fi
  # Fallback when CPU detection tools are unavailable.
  echo 4
}

MAX_PARALLEL="${MAX_PARALLEL:-$(detect_max_parallel)}"

# Remove positional start/end from forwarded args when explicitly provided.
if [[ $# -ge 1 ]]; then
  EXTRA_ARGS=("${EXTRA_ARGS[@]:1}")
fi
if [[ $# -ge 2 ]]; then
  EXTRA_ARGS=("${EXTRA_ARGS[@]:1}")
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/run_all_tender_ids.sh [START_ID] [END_ID] [extra CLI args...]

Examples:
  scripts/run_all_tender_ids.sh
  scripts/run_all_tender_ids.sh 000001 000500
  scripts/run_all_tender_ids.sh 000001 000100 --skip-award --rate-limit-seconds 1.5

Environment variables:
  OUTPUT_ROOT          Root output directory (default: <repo>/downloads/batch_runs)
  LOG_DIR              Log directory (default: <OUTPUT_ROOT>/logs)
  RATE_LIMIT_SECONDS   CLI rate limit between requests (default: 2.0)

Notes:
  - IDs are always processed as 6-digit values.
  - Missing/non-existent tenders are logged and processing continues.
EOF
}

validate_id() {
  local value="$1"
  [[ "$value" =~ ^[0-9]{6}$ ]]
}

if [[ "${START_ID}" == "-h" || "${START_ID}" == "--help" ]]; then
  usage
  exit 0
fi

if ! validate_id "${START_ID}"; then
  echo "Invalid START_ID: ${START_ID} (expected exactly 6 digits)." >&2
  exit 2
fi

if ! validate_id "${END_ID}"; then
  echo "Invalid END_ID: ${END_ID} (expected exactly 6 digits)." >&2
  exit 2
fi

if ! [[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid MAX_PARALLEL: ${MAX_PARALLEL} (expected a positive integer)." >&2
  exit 2
fi

START_NUM=$((10#${START_ID}))
END_NUM=$((10#${END_ID}))

if (( START_NUM > END_NUM )); then
  echo "START_ID must be <= END_ID." >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

SUMMARY_FILE="${LOG_DIR}/summary.txt"
SUCCESS_IDS_FILE="${LOG_DIR}/success_ids.txt"
FAILED_IDS_FILE="${LOG_DIR}/failed_or_missing_ids.txt"

: >"${SUMMARY_FILE}"
: >"${SUCCESS_IDS_FILE}"
: >"${FAILED_IDS_FILE}"

TOTAL=0
SUCCESS=0
FAILED_OR_MISSING=0

trap 'echo; echo "Interrupted. Partial progress saved in ${LOG_DIR}."; exit 130' INT TERM

echo "Starting batch run from ${START_ID} to ${END_ID}" | tee -a "${SUMMARY_FILE}"
echo "Output root: ${OUTPUT_ROOT}" | tee -a "${SUMMARY_FILE}"
echo "Rate limit: ${RATE_LIMIT_SECONDS}s" | tee -a "${SUMMARY_FILE}"
echo "Parallel workers: ${MAX_PARALLEL}" | tee -a "${SUMMARY_FILE}"

run_tender() {
  local tender_id="$1"
  local run_dir="${OUTPUT_ROOT}/${tender_id}"
  local run_log="${run_dir}/run.log"

  mkdir -p "${run_dir}"

  if uv run "${WORK_DIR}/scripts/telangana_tender_cli.py" \
      --tender-id "${tender_id}" \
      --write-json \
      --download \
      --unzip \
      --output-dir "${run_dir}" \
      --rate-limit-seconds "${RATE_LIMIT_SECONDS}" \
      "${EXTRA_ARGS[@]}" >"${run_log}" 2>&1; then
    echo "${tender_id}" >>"${SUCCESS_IDS_FILE}"
  else
    echo "${tender_id}" >>"${FAILED_IDS_FILE}"
    echo "  -> skipped (missing tender or request failure). See ${run_log}"
  fi
}

for (( n=START_NUM; n<=END_NUM; n++ )); do
  TENDER_ID="$(printf "%06d" "${n}")"

  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n || true
  done

  TOTAL=$((TOTAL + 1))

  echo "[${TOTAL}] Queued tender ${TENDER_ID}"
  run_tender "${TENDER_ID}" &
done

wait

SUCCESS=$(wc -l <"${SUCCESS_IDS_FILE}")
FAILED_OR_MISSING=$(wc -l <"${FAILED_IDS_FILE}")

{
  echo ""
  echo "Completed batch run"
  echo "Total processed: ${TOTAL}"
  echo "Success: ${SUCCESS}"
  echo "Failed or missing: ${FAILED_OR_MISSING}"
  echo "Success IDs file: ${SUCCESS_IDS_FILE}"
  echo "Failed/missing IDs file: ${FAILED_IDS_FILE}"
} | tee -a "${SUMMARY_FILE}"
