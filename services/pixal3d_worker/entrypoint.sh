#!/bin/sh
# pixal3d-worker container entry: the stage runner (as before) plus, unless STUDIO_PIXAL3D_RESIDENT=0, the
# resident Pixal3D session in a restart loop (it exits on purpose after a failed or cancelled job).
set -u
if [ "${STUDIO_PIXAL3D_RESIDENT:-1}" != "0" ]; then
  (
    while true; do
      python -m pixal3d_worker.serve
      echo "[entrypoint] resident session exited with $?; restarting in 5 s"
      sleep 5
    done
  ) &
fi
exec python -m runner.runner --port 8702
