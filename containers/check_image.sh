#!/usr/bin/env bash
# Check that the Apptainer image was built from the uv.lock and
# containers/afabench.def in this checkout.
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

# Show Apptainer's own error when the image does not run at all, for example
# after an incomplete copy.
if ! apptainer exec "${image}" true; then
    echo "Apptainer cannot run ${image}, see the error above. If the copy" \
        "was interrupted, its size differs from the built image." >&2
    exit 1
fi

expected="$(cat uv.lock containers/afabench.def | sha256sum | cut -d' ' -f1)"
# Images built before the definition was part of the hash have no such file.
actual="$(apptainer exec "${image}" cat /opt/afabench/image.sha256 2>/dev/null \
    || echo "none (an older image)")"
if [[ "${expected}" != "${actual}" ]]; then
    echo "${image} was built from a different uv.lock or" \
        "containers/afabench.def." >&2
    echo "  checkout: ${expected}" >&2
    echo "  image:    ${actual}" >&2
    echo "Rebuild it with containers/build.sh and copy it over." >&2
    exit 1
fi
echo "${image} matches uv.lock and containers/afabench.def" \
    "(${expected:0:12})."
