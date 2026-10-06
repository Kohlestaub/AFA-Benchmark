#!/usr/bin/env bash
# Check that the Apptainer image was built from the uv.lock in this checkout.
#
# Usage: containers/check_image.sh [image]   (default: containers/afabench.sif)
# Exits with status 1 when the image is missing or out of date.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

image="${1:-containers/afabench.sif}"
if [[ ! -e "${image}" ]]; then
    echo "No image at ${image}. Build it with containers/build.sh." >&2
    exit 1
fi

expected="$(sha256sum uv.lock | cut -d' ' -f1)"
actual="$(apptainer exec "${image}" cat /opt/afabench/uv.lock.sha256)"
if [[ "${expected}" != "${actual}" ]]; then
    echo "${image} was built from a different uv.lock." >&2
    echo "  checkout: ${expected}" >&2
    echo "  image:    ${actual}" >&2
    echo "Rebuild it with containers/build.sh and copy it over." >&2
    exit 1
fi
echo "${image} matches uv.lock (${expected:0:12})."
