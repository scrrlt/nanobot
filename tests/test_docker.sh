#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

IMAGE_NAME="nanobot-test"
CONTAINER_NAME="nanobot-test-run"

cleanup() {
    echo ""
    echo "=== Cleanup ==="
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker rmi -f nanobot-test-onboarded 2>/dev/null || true
    docker rmi -f "$IMAGE_NAME" 2>/dev/null || true
    echo "Done."
}

trap cleanup EXIT

echo "=== Building Docker image ==="
docker build -t "$IMAGE_NAME" .

echo ""
echo "=== Running 'nanobot onboard' ==="
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker run --name "$CONTAINER_NAME" "$IMAGE_NAME" onboard

echo ""
echo "=== Running 'nanobot status' ==="
STATUS_OUTPUT=$(docker commit "$CONTAINER_NAME" nanobot-test-onboarded > /dev/null && \
    docker run --rm nanobot-test-onboarded status 2>&1) || true

echo "$STATUS_OUTPUT"

echo ""
echo "=== Validating output ==="
PASS=true

check() {
    if echo "$STATUS_OUTPUT" | grep -qi "$1"; then
        echo "  PASS: found '$1'"
    else
        echo "  FAIL: missing '$1'"
        PASS=false
    fi
}

check "Nanobot Status"
check "Config:"
check "Workspace:"
check "Model:"
check "OpenRouter API:"
check "Anthropic API:"
check "OpenAI API:"

echo ""
if $PASS; then
    echo "=== All checks passed ==="
else
    echo "=== Some checks FAILED ==="
    exit 1
fi
