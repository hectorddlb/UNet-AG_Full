# source env.sh  — from the repository root
# Makes importable:  unet_ag.*        (proposed model, code/unet_ag)
#                    schiffner_das.*  (Fd-F port, code/schiffner_das)
#                    drus.*           (DRUS replica, code/drus)
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${_ROOT}/code:${PYTHONPATH:-}"
# Path of the raw datasets (PICMUS, Field II), which do NOT live in the repo.
export THESIS_DATASETS="${THESIS_DATASETS:-${_ROOT}/data}"
echo "PYTHONPATH: ${_ROOT}/code"
echo "THESIS_DATASETS: ${THESIS_DATASETS}"
