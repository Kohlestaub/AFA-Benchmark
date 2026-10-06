#!/usr/bin/env bash
# Build the AFABench Apptainer image from containers/afabench.def.
#
# Run this on a machine where you can build images (root through sudo, or a
# working --fakeroot setup), for example your own Linux machine, and copy the
# result to the cluster. The image name contains the first 12 characters of
# the sha256 of uv.lock, so an existing image is reused until the
# dependencies change. containers/afabench.sif is a symlink to the newest one.
#
# Usage (from anywhere inside the repository):
#   containers/build.sh                       # uses sudo
#   APPTAINER_FAKEROOT=1 containers/build.sh  # uses --fakeroot instead
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

lock_hash="$(sha256sum uv.lock | cut -c1-12)"
image="containers/afabench-${lock_hash}.sif"

if [[ -e "${image}" ]]; then
    echo "${image} already exists, uv.lock has not changed since it was built."
elif [[ "${APPTAINER_FAKEROOT:-0}" == "1" ]]; then
    apptainer build --fakeroot "${image}" containers/afabench.def
else
    sudo apptainer build "${image}" containers/afabench.def
fi

ln -sfn "$(basename "${image}")" containers/afabench.sif

cat <<EOF

Image: ${image} ($(du -h "${image}" | cut -f1))
Copy it to the cluster and point the symlink at it, for example:
  rsync -P ${image} minerva:<repo>/containers/
  ssh minerva 'cd <repo> && ln -sfn $(basename "${image}") containers/afabench.sif'
EOF
