#!/usr/bin/env bash
#
# Download the two OpenCV Zoo ONNX models the face backend needs:
#
#   face_detection_yunet_2023mar.onnx   ~230 KB   answers "is there a face here?"
#   face_recognition_sface_2021dec.onnx ~37 MB    answers "whose face is it?"
#
# Both are plain ONNX files loaded through cv2.dnn, so face recognition needs no onnxruntime
# and no insightface.  They are not committed (see .gitignore); run this once per machine.
#
# IMPORTANT -- these two files live in Git LFS upstream.  `raw.githubusercontent.com` therefore
# returns a ~130-byte *pointer* file, not the model, and the usual ghproxy mirrors faithfully
# pass that pointer through.  The real bytes are only served from the LFS media endpoint,
# `media.githubusercontent.com/media/...`, which is what this script uses.  A download that
# yields a tiny file is reported as a pointer, not silently accepted.
#
# SFace is ~37 MB and this script was developed on a network that reached the media endpoint at
# only ~60 KB/s, so a full first run can take ~10 minutes.  The download resumes: interrupt it
# (Ctrl-C) or let it time out, then re-run and it continues from where it stopped.
#
# Usage:
#   scripts/fetch_models.sh              # skip files that are already present
#   scripts/fetch_models.sh --force      # re-download even if present
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"

FORCE=0
for arg in "$@"; do
    case "${arg}" in
        --force|-f) FORCE=1 ;;
        -h|--help)
            sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
    esac
done

# Upstream paths inside opencv/opencv_zoo.
YUNET_PATH="models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_PATH="models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# The LFS media endpoint is the only one that serves the actual bytes.  The two mirrors are
# tried after it because a network that reaches GitHub through a proxy may still reach them;
# they return 403/404 on a direct network, which is harmless since the loop just moves on.
BLOB_BASES=(
    "https://media.githubusercontent.com/media"
    "https://gh-proxy.com/https://media.githubusercontent.com/media"
    "https://ghproxy.net/https://media.githubusercontent.com/media"
)

# Exact byte sizes, used as the integrity check: a truncated download or an LFS pointer would
# otherwise only surface later as an opaque `cv2.error` when the model is loaded.
YUNET_BYTES=232589
SFACE_BYTES=38696353

# Abort a stalled transfer quickly so the shell-level retry loop below can resume it.  curl's
# own ``--retry`` is deliberately NOT used: on a ranged request it can restart the transfer from
# zero, which throws away the bytes already on disk.  This network delivers a few MB from the
# media endpoint and then stalls, so a single curl call rarely finishes a 37 MB file.
CURL_OPTS=(-fL --connect-timeout 15 --speed-limit 1024 --speed-time 20)
MAX_ATTEMPTS=20

log() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

have_size() {
    local path="$1" want="$2"
    [[ -f "${path}" ]] && [[ "$(stat -c '%s' "${path}" 2>/dev/null || echo 0)" == "${want}" ]]
}

file_size() {
    stat -c '%s' "$1" 2>/dev/null || echo 0
}

is_lfs_pointer() {
    # An LFS pointer is a short text file starting with the LFS spec version line.
    [[ "$(head -c 20 "$1" 2>/dev/null)" == "version https://git-l"* ]]
}

fetch() {
    # fetch <relative-path> <expected-bytes> <destination>
    local rel="$1" want="$2" dest="$3"

    if [[ "${FORCE}" -eq 0 ]] && have_size "${dest}" "${want}"; then
        log "skip   $(basename "${dest}") (already present, ${want} bytes)"
        return 0
    fi

    # A leftover .part file is resumed rather than discarded, which is what makes a slow or
    # interrupted download recoverable.
    local tmp="${dest}.part"

    for base in "${BLOB_BASES[@]}"; do
        local url="${base}/opencv/opencv_zoo/main/${rel}"

        # Retry in the shell, always resuming from whatever is already on disk.
        local attempt
        for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
            if [[ -s "${tmp}" ]]; then
                log "resume $(basename "${dest}") from $(file_size "${tmp}") bytes" \
                    "(attempt ${attempt}/${MAX_ATTEMPTS}) via ${base}"
            else
                log "fetch  ${url} (attempt ${attempt}/${MAX_ATTEMPTS})"
            fi
            curl "${CURL_OPTS[@]}" -C - -o "${tmp}" "${url}" && break
            sleep 2
        done

        [[ -f "${tmp}" ]] || continue

        if is_lfs_pointer "${tmp}"; then
            log "warn   got an LFS pointer file (not the model); trying next mirror"
            rm -f "${tmp}"
            continue
        fi

        local got
        got="$(file_size "${tmp}")"
        if [[ "${got}" == "${want}" ]]; then
            mv "${tmp}" "${dest}"
            log "ok     $(basename "${dest}") (${got} bytes)"
            return 0
        fi

        if (( got > want )); then
            # Longer than expected means the file is not what we think it is; resuming it would
            # only grow the mistake.
            rm -f "${tmp}"
        fi
        log "warn   incomplete: ${got}/${want} bytes; trying next mirror"
    done

    die "could not finish downloading ${rel}.
     Re-run this script to resume from $(file_size "${tmp}") bytes, or fetch it manually:
       https://media.githubusercontent.com/media/opencv/opencv_zoo/main/${rel}
     and save it as ${dest} (expected ${want} bytes)."
}

command -v curl >/dev/null 2>&1 || die "curl is required (sudo apt install curl)"

mkdir -p "${MODELS_DIR}"

fetch "${YUNET_PATH}" "${YUNET_BYTES}" "${MODELS_DIR}/face_detection_yunet_2023mar.onnx"
fetch "${SFACE_PATH}" "${SFACE_BYTES}" "${MODELS_DIR}/face_recognition_sface_2021dec.onnx"

log ""
log "done. Face models are in ${MODELS_DIR}"
log "check the backend with: qwen25vl-agent doctor"
