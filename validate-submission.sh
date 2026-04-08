#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
# Usage: ./validate-submission.sh <ping_url> [repo_dir]

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  exit 1
fi

REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"
PING_URL="${PING_URL%/}"
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }

printf "\n${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"

# Step 1: Ping
log "${BOLD}Step 1/3: Pinging HF Space${NC}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>/dev/null || printf "000")
if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE"
fi

# Step 2: Docker build
log "${BOLD}Step 2/3: Docker build${NC}"
if command -v docker &>/dev/null; then
  docker build "$REPO_DIR" && pass "Docker build succeeded" || fail "Docker build failed"
else
  fail "docker not found"
fi

# Step 3: openenv validate
log "${BOLD}Step 3/3: openenv validate${NC}"
if command -v openenv &>/dev/null; then
  (cd "$REPO_DIR" && openenv validate) && pass "openenv validate passed" || fail "openenv validate failed"
else
  fail "openenv not found"
fi

printf "\n${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  $PASS/3 checks passed${NC}\n"
printf "${BOLD}========================================${NC}\n"
