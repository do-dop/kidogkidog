#!/usr/bin/env bash
# 실습 VM의 loopback 메트릭 포트(worker 8002, RabbitMQ 15692)를 로컬로 연결한다.
set -euo pipefail
exec gcloud compute ssh "${LAB_VM:-lab-worker}" \
  --zone "${LAB_ZONE:-asia-northeast3-a}" --project "${GCP_PROJECT:-kidogkidog-509506}" \
  -- -N -L 18002:127.0.0.1:8002 -L 15692:127.0.0.1:15692
