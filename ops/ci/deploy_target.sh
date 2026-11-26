#!/usr/bin/env bash
# 지금 체크아웃한 커밋(HEAD)으로 한 서비스를 배포해야 하는지 판단한다.
#
# 사용: deploy_target.sh <api|web> <지금 트래픽을 받는 리비전의 커밋 SHA(없으면 빈 값)> <이벤트 이름>
# 출력: "deploy=true|false" 와 "reason=..." (GITHUB_OUTPUT 형식)
#
# 직전 커밋과의 차이가 아니라 "지금 서비스 중인 커밋"과의 차이를 본다. 그래서 중간 배포가
# 취소되거나 건너뛰어도 다음 배포가 빠진 변경을 함께 내보낸다.
set -euo pipefail

component=${1:?component가 필요합니다}
deployed=${2-}
event=${3:?이벤트 이름이 필요합니다}

case "$component" in
  api) paths='^(api/|db/|pipeline/|requirements\.txt$|api/Dockerfile$)' ;;
  web) paths='^ui-react/' ;;
  *) echo "알 수 없는 component: $component" >&2; exit 2 ;;
esac

decide() {
  echo "deploy=$1"
  echo "reason=$2"
  exit 0
}

# 수동 실행은 항상 다시 배포한다.
[[ "$event" == "workflow_dispatch" ]] && decide true "수동 실행"

# 커밋을 알 수 없으면(라벨 없는 이전 리비전, 서비스 없음, 저장소에 없는 SHA) 비교할 수 없으므로 배포한다.
if [[ ! "$deployed" =~ ^[0-9a-f]{40}$ ]] || ! git cat-file -e "${deployed}^{commit}" 2>/dev/null; then
  decide true "서비스 중인 커밋을 알 수 없음"
fi

head=$(git rev-parse HEAD)
[[ "$deployed" == "$head" ]] && decide false "이미 이 커밋을 서비스 중"
if git merge-base --is-ancestor "$head" "$deployed"; then
  decide false "더 새로운 커밋(${deployed:0:7})을 서비스 중"
fi

# 서비스 중인 커밋이 HEAD의 조상이 아니어도(다른 브랜치를 올린 경우) 두 트리의 차이로 판단한다.
# --no-renames: 파일이 경로 밖으로 옮겨져도 원래 경로가 목록에 남아야 한다.
changed=$(git diff --name-only --no-renames "$deployed" "$head")
if grep -Eq "$paths" <<< "$changed"; then
  decide true "${deployed:0:7} 이후 ${component} 변경 있음"
fi
decide false "${deployed:0:7} 이후 ${component} 변경 없음"
