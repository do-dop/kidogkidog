#!/usr/bin/env bash
set -euo pipefail

# Keep this process running while a local API or worker accesses GCP ChromaDB.
# Bind only to loopback; do not expose ChromaDB on the laptop's LAN interface.
exec gcloud compute ssh "${CHROMA_VM:-kidog-chroma}" \
  --project "${GCP_PROJECT_ID:-kidogkidog-509506}" \
  --zone "${CHROMA_ZONE:-asia-northeast3-a}" \
  -- -N \
  -L "127.0.0.1:${CHROMA_PORT:-18001}:127.0.0.1:8000" \
  -o ExitOnForwardFailure=yes \
  -o ConnectTimeout=15 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3
