#!/bin/bash
# =============================================================================
# submit_imerg_jobs.sh
# Submit a PBS array job that extracts IMERG precipitation features.
# One array task per month; each task writes one CSV.
#
# Detection is independent of the swath: every feature above the threshold is
# found and recorded, and the swath only annotates it. `cross_swath_extent_km`
# lets you re-evaluate any swath width from the CSVs without reprocessing.
#
# RAW HDF5 INPUT: reads half-hourly granules straight from /data05/IMERG_GPM
# (READ-ONLY) instead of the monthly 6-hourly netCDF. MODE selects the
# accumulation:
#   halfhour -- every granule is a timestep, 48/day  (1344 per February)
#   hourly   -- mean of the :00 and :30 granules,    (672 per February)
# /Grid/precipitation is already a rate in mm/hr, so THRESHOLD means the same
# thing in both modes and hourly is a plain mean of two rates.
#
# Output goes to a per-mode folder (output_<mode>_inswathcounted/) so the two
# runs cannot collide with each other or with the completed
# output_inswathcounted/. CSV file names are unchanged (imerg_features_YYYYMM.csv).
#
# LONGITUDE: the granules are -180..180 but the old netCDF was 0..360. The
# swath tiling is anchored at origin lon 180.0, so the convention decides where
# the seams fall. LON_CONVENTION defaults to 0-360 to stay comparable with
# output_inswathcounted/. Do not change it for a comparison run.
#
# Update gridfeatures in the conda env first, or the worker raises
# AttributeError in build_statistics:
#   conda activate imerg_precipitation_features
#   pip install --force-reinstall --no-deps \
#     "git+https://github.com/pedroanguloumana/General_Gridded_Feature_Extraction.git"
#   python -c "from gridfeatures import stats; assert hasattr(stats,'swath_edge_pixels_in_dominant')"
#
# Usage:
#   bash submit_imerg_jobs.sh                       # Feb 2015-2022, halfhour
#   MODE=hourly bash submit_imerg_jobs.sh           # hourly instead
#   bash submit_imerg_jobs.sh 2015 2022 02          # explicit years + month
#   DRY_RUN=1 bash submit_imerg_jobs.sh             # print the PBS script, do not submit
#   SMOKE=4  bash submit_imerg_jobs.sh 2016 2016    # 4 timesteps only, one month
# =============================================================================

set -euo pipefail

# ── User-defined parameters ─────────────────────────────────────────────────
YEAR_START="${1:-2015}"
YEAR_END="${2:-2022}"
MONTH="${3:-02}"

MODE="${MODE:-halfhour}"               # halfhour | hourly
LON_CONVENTION="${LON_CONVENTION:-0-360}"   # 0-360 matches output_inswathcounted/

case "${MODE}" in
    halfhour|hourly) ;;
    *) echo "ERROR: MODE must be halfhour or hourly, got '${MODE}'"; exit 1 ;;
esac

LAT_MIN="${LAT_MIN:--20}"
LAT_MAX="${LAT_MAX:-20}"
THRESHOLD="${THRESHOLD:-1.0}"          # mm/hr
MIN_SIZE="${MIN_SIZE:-5}"              # pixels
CONNECTIVITY="${CONNECTIVITY:-2}"      # 8-connectivity
SWATH_WIDTH_KM="${SWATH_WIDTH_KM:-245}"
SWATH_ANGLE_DEG="${SWATH_ANGLE_DEG:-65}"
ORIGIN_LAT="${ORIGIN_LAT:-0.0}"
ORIGIN_LON="${ORIGIN_LON:-180.0}"

CONDA_ENV="${CONDA_ENV:-imerg_precipitation_features}"
# Measured ~3.0 s/timestep: halfhour February = 1344 steps ~= 1.1 h, hourly =
# 672 steps ~= 0.6 h. 04:00:00 leaves >3x headroom on the slower mode.
WALLTIME="${WALLTIME:-04:00:00}"
# Pin a compute node. tropics00 is the job-control node -- never run work there.
EXEC_HOST="${EXEC_HOST:-tropics02}"
PBS_SELECT="${PBS_SELECT:-select=1:ncpus=1:mem=16gb:host=${EXEC_HOST}}"
JOB_NAME="${JOB_NAME:-imerg_feat_${MODE}}"   # mode in the name keeps logs apart

DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"                    # >0 => pass --max-timesteps SMOKE

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT="/home1/pedro/Projects/imerg_precipitation_features"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/imerg_features.py"
# Symlink -> /data05/IMERG_GPM. READ-ONLY: never written to.
ARCHIVE="${ARCHIVE:-${PROJECT}/data/imerg_hhr}"
# Per-mode output folder: keeps halfhour and hourly apart, and keeps both away
# from the completed output_inswathcounted/ so skip-if-exists cannot reuse a
# CSV from a different mode or an older schema.
OUTDIR="${PROJECT}/output_${MODE}_inswathcounted"
LOGDIR="${PROJECT}/logs"

# A smoke run must not drop a truncated CSV into the real output dir -- the
# skip-if-exists check would then silently skip that month on the real run.
[[ "${SMOKE:-0}" -gt 0 ]] && OUTDIR="${PROJECT}/output_smoke_${MODE}_inswathcounted"

[[ -d "${ARCHIVE}" ]] || { echo "ERROR: archive not found: ${ARCHIVE}"; exit 1; }

[[ -f "${WORKER}" ]] || { echo "ERROR: worker not found: ${WORKER}"; exit 1; }
mkdir -p "${OUTDIR}" "${LOGDIR}"

# ── Discover months, skip already-processed, refuse incomplete ───────────────
# The worker re-checks completeness authoritatively and raises before doing any
# work; this pre-flight just avoids queueing a task that is certain to die.
# Granules are counted by the date INSIDE the filename across the neighbouring
# year directories, because the archive misfiles some (2020/ holds three
# 20210301 granules), and *.HDF5.1 re-download duplicates are excluded.
TODO=()
for (( Y=YEAR_START; Y<=YEAR_END; Y++ )); do
    YYYYMM="${Y}${MONTH}"
    OUT="${OUTDIR}/imerg_features_${YYYYMM}.csv"

    if [[ -f "${OUT}" && "${SMOKE}" -eq 0 ]]; then
        echo "SKIP (output exists): $(basename "${OUT}")"
        continue
    fi

    DAYS=$(date -d "${Y}-${MONTH}-01 +1 month -1 day" +%d)
    EXPECTED=$(( 10#${DAYS} * 48 ))
    # Only search year directories that exist. Under `set -o pipefail` a find
    # over a missing directory returns non-zero and would abort the whole
    # script -- which is what happens at the ends of the archive, where a
    # neighbouring year (2013, 2024) is simply not there.
    SEARCH=()
    for YY in $((Y-1)) "${Y}" $((Y+1)); do
        [[ -d "${ARCHIVE}/${YY}" ]] && SEARCH+=("${ARCHIVE}/${YY}")
    done
    if [[ ${#SEARCH[@]} -eq 0 ]]; then
        echo "WARN: ${YYYYMM} no archive directory, skipping"
        continue
    fi
    FOUND=$(find "${SEARCH[@]}" \
                 -maxdepth 1 -name "3B-HHR.MS.MRG.3IMERG.${YYYYMM}*.HDF5" -printf '%f\n' \
                 2>/dev/null | sort -u | wc -l)

    if [[ "${FOUND}" -ne "${EXPECTED}" ]]; then
        echo "WARN: ${YYYYMM} incomplete (${FOUND}/${EXPECTED} granules), skipping"
        continue
    fi
    TODO+=("${YYYYMM}")
done

N=${#TODO[@]}
if [[ ${N} -eq 0 ]]; then
    echo "Nothing to do."
    exit 0
fi

FILELIST="${LOGDIR}/filelist_${JOB_NAME}_$(date +%Y%m%d_%H%M%S).txt"
printf "%s\n" "${TODO[@]}" > "${FILELIST}"

echo "─────────────────────────────────────────────────────────────"
echo " months to process : ${N}   (${TODO[0]} .. ${TODO[-1]})"
echo " exec host         : ${EXEC_HOST}"
echo " archive           : ${ARCHIVE} -> $(readlink -f "${ARCHIVE}")"
echo " mode              : ${MODE} ($([[ "${MODE}" == hourly ]] && echo "24" || echo "48") timesteps/day)"
echo " lon convention    : ${LON_CONVENTION}"
echo " band              : ${LAT_MIN} .. ${LAT_MAX}"
echo " swath             : ${SWATH_WIDTH_KM} km @ ${SWATH_ANGLE_DEG} deg, origin (${ORIGIN_LAT}, ${ORIGIN_LON})"
echo " detection         : > ${THRESHOLD} mm/hr, min ${MIN_SIZE} px, connectivity ${CONNECTIVITY}"
echo " output            : ${OUTDIR}"
echo " logs              : ${LOGDIR}"
[[ "${SMOKE}" -gt 0 ]] && echo " SMOKE TEST        : first ${SMOKE} timesteps only"
echo "─────────────────────────────────────────────────────────────"

SMOKE_ARG=""
[[ "${SMOKE}" -gt 0 ]] && SMOKE_ARG="--max-timesteps ${SMOKE} --overwrite"

# ── Build the PBS array script ──────────────────────────────────────────────
read -r -d '' PBS_SCRIPT <<'OUTER_EOF' || true
#!/bin/bash
#PBS -N __JOB_NAME__
#PBS -l __PBS_SELECT__
#PBS -l walltime=__WALLTIME__
# Do NOT rely on PBS to stage stdout back. From a compute node it copies the
# spool file to the submit host and, on this cluster, that copy fails: the logs
# pile up unread in tropics02:/var/spool/pbs/undelivered/. LOGDIR is on shared
# NFS, so redirect there ourselves and let PBS discard its own capture.
#PBS -o /dev/null
#PBS -e /dev/null

IDX="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-0}}"
JOBNUM=$(echo "${PBS_JOBID:-local}" | cut -d. -f1 | cut -d'[' -f1)
mkdir -p "__LOGDIR__"
exec > "__LOGDIR__/__JOB_NAME__.${JOBNUM}.${IDX}.log" 2>&1

set -euo pipefail

# One task = one month. Keep BLAS single-threaded: we get parallelism from the array.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# /data05 may be NFS, where h5py/HDF5 can refuse even a read-only open while it
# tries to take a file lock. We never write to the archive, so disable locking.
export HDF5_USE_FILE_LOCKING=FALSE

# Refuse to run on the job-control node, whatever PBS decided.
if [[ "$(hostname -s)" == "tropics00" ]]; then
    echo "ERROR: scheduled onto tropics00 (job-control node). Refusing to run."
    exit 1
fi

IDX="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-0}}"
YYYYMM=$(sed -n "$((IDX + 1))p" "__FILELIST__")
if [[ -z "${YYYYMM}" ]]; then
    echo "ERROR: no entry at index ${IDX} in __FILELIST__"
    exit 1
fi

set +u
source /usr/local/anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate __CONDA_ENV__
set -u

echo "host=$(hostname) task=${IDX} month=${YYYYMM} mode=__MODE__ start=$(date -Is)"

python "__WORKER__" \
    "${YYYYMM}" \
    "__OUTDIR__/imerg_features_${YYYYMM}.csv" \
    --archive "__ARCHIVE__" \
    --mode __MODE__ \
    --lon-convention __LON_CONVENTION__ \
    --lat-min __LAT_MIN__ --lat-max __LAT_MAX__ \
    --threshold __THRESHOLD__ --min-size __MIN_SIZE__ \
    --connectivity __CONNECTIVITY__ \
    --swath-width-km __SWATH_WIDTH_KM__ --swath-angle-deg __SWATH_ANGLE_DEG__ \
    --origin-lat __ORIGIN_LAT__ --origin-lon __ORIGIN_LON__ \
    __SMOKE_ARG__

echo "done=$(date -Is)"
OUTER_EOF

subst() { PBS_SCRIPT="${PBS_SCRIPT//$1/$2}"; }
subst __JOB_NAME__       "${JOB_NAME}"
subst __PBS_SELECT__     "${PBS_SELECT}"
subst __WALLTIME__       "${WALLTIME}"
subst __LOGDIR__         "${LOGDIR}"
subst __FILELIST__       "${FILELIST}"
subst __CONDA_ENV__      "${CONDA_ENV}"
subst __WORKER__         "${WORKER}"
subst __ARCHIVE__        "${ARCHIVE}"
subst __MODE__           "${MODE}"
subst __LON_CONVENTION__ "${LON_CONVENTION}"
subst __OUTDIR__         "${OUTDIR}"
subst __LAT_MIN__        "${LAT_MIN}"
subst __LAT_MAX__        "${LAT_MAX}"
subst __THRESHOLD__      "${THRESHOLD}"
subst __MIN_SIZE__       "${MIN_SIZE}"
subst __CONNECTIVITY__   "${CONNECTIVITY}"
subst __SWATH_WIDTH_KM__ "${SWATH_WIDTH_KM}"
subst __SWATH_ANGLE_DEG__ "${SWATH_ANGLE_DEG}"
subst __ORIGIN_LAT__     "${ORIGIN_LAT}"
subst __ORIGIN_LON__     "${ORIGIN_LON}"
subst __SMOKE_ARG__      "${SMOKE_ARG}"

# PBS rejects a single-element array (`-J 0-0`). Submit a plain job in that case
# and hand it the array index through the environment instead.
if [[ ${N} -eq 1 ]]; then
    QSUB_ARGS=(-v "PBS_ARRAY_INDEX=0")
else
    QSUB_ARGS=(-J "0-$((N - 1))")
fi

if [[ "${DRY_RUN}" != "0" ]]; then
    echo "── DRY RUN: PBS script below, nothing submitted ──"
    echo "${PBS_SCRIPT}"
    echo "── would submit: qsub ${QSUB_ARGS[*]} ──"
    exit 0
fi

JOBID=$(echo "${PBS_SCRIPT}" | qsub "${QSUB_ARGS[@]}")
echo "submitted job: ${JOBID}"
echo "  monitor : qstat -t ${JOBID}"
echo "  logs    : ${LOGDIR}/${JOB_NAME}.o<jobid>.<index>   (written when a task finishes)"
echo "  results : ${OUTDIR}/imerg_features_YYYYMM.csv"