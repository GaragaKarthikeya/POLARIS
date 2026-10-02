#!/usr/bin/env bash
# Install POLARIS into a checkout of COSM's public artifact (doi:10.5281/zenodo.19660293) and build the simulator.
# usage: COSM_ROOT=/path/to/cosm-artifact scripts/setup.sh [make-jobs]
set -euo pipefail
: "${COSM_ROOT:?set COSM_ROOT to the directory of the COSM artifact}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
SIM="$COSM_ROOT/simulations/simulator"
# 1) POLARIS harness = COSM's Simulator.py + a small patch (COSM's scheduling code is not modified)
cp "$SIM/Simulator.py" "$SIM/Simulator_polaris.py"
patch --quiet "$SIM/Simulator_polaris.py" < "$HERE/harness/simulator_polaris.patch"
cp "$HERE/polaris/polaris_agent.py" "$SIM/polaris_agent.py"
# 2) rotated copies of the CPU traces (each run can start the CPU trace at a different point)
python3 "$HERE/scripts/make_rotations.py"
# 3) build the unmodified COSM simulator
cd "$COSM_ROOT" && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j"${1:-8}"
echo "POLARIS installed into $COSM_ROOT"
