#!/bin/bash
# =============================================================================
# Oracle Cloud Free Tier — GitHub Actions Self-hosted Runner 셋업 스크립트
# Ubuntu 22.04 ARM (VM.Standard.A1.Flex) 기준
#
# 사용법:
#   chmod +x setup_oracle_runner.sh
#   ./setup_oracle_runner.sh <GITHUB_REPO_URL> <RUNNER_TOKEN>
#
# RUNNER_TOKEN 발급:
#   GitHub → 레포 → Settings → Actions → Runners → New self-hosted runner
#   → Linux/ARM64 선택 후 토큰 복사
# =============================================================================

set -euo pipefail

REPO_URL="${1:-}"
RUNNER_TOKEN="${2:-}"

if [ -z "$REPO_URL" ] || [ -z "$RUNNER_TOKEN" ]; then
  echo "사용법: $0 <GITHUB_REPO_URL> <RUNNER_TOKEN>"
  echo "예시:   $0 https://github.com/Greedyguy/QuantInvest ghp_xxxxx"
  exit 1
fi

echo "======================================================"
echo "  Oracle Cloud Runner 셋업 시작"
echo "  레포: $REPO_URL"
echo "======================================================"

# ── 1. 시스템 패키지 업데이트 ──────────────────────────────
echo ""
echo "[1/6] 시스템 패키지 업데이트..."
sudo apt-get update -q
sudo apt-get install -y -q \
  python3.11 python3.11-venv python3.11-dev python3-pip \
  git curl wget unzip jq \
  build-essential libssl-dev libffi-dev

# Python 기본 설정
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

echo "  ✅ Python $(python3 --version)"

# ── 2. KST 타임존 설정 ─────────────────────────────────────
echo ""
echo "[2/6] 타임존 → KST (Asia/Seoul) 설정..."
sudo timedatectl set-timezone Asia/Seoul
echo "  ✅ $(timedatectl | grep 'Time zone')"

# ── 3. GitHub Actions Runner 설치 ─────────────────────────
echo ""
echo "[3/6] GitHub Actions Runner 설치..."
RUNNER_VERSION="2.316.1"
RUNNER_DIR="$HOME/actions-runner"
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

curl -sO -L \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz"
tar xzf "actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz"
rm "actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz"

echo "  ✅ Runner 다운로드 완료 (v${RUNNER_VERSION})"

# ── 4. Runner 등록 ─────────────────────────────────────────
echo ""
echo "[4/6] Runner 등록..."
./config.sh \
  --url "$REPO_URL" \
  --token "$RUNNER_TOKEN" \
  --name "oracle-kr-trader" \
  --labels "self-hosted,oracle,kr-trader" \
  --work "_work" \
  --unattended

echo "  ✅ Runner 등록 완료"

# ── 5. Python 패키지 사전 설치 (매번 pip install 방지) ─────
echo ""
echo "[5/6] Python 패키지 사전 설치..."
VENV_DIR="$HOME/trader-venv"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# requirements.txt 경로 (runner work 디렉토리에 아직 없으므로 직접 설치)
pip install --upgrade pip -q
pip install -q \
  pandas numpy matplotlib \
  pykrx pyarrow fastparquet \
  requests python-telegram-bot \
  tqdm scikit-learn

echo "  ✅ 패키지 설치 완료"
deactivate

# ── 6. systemd 서비스 등록 (재부팅 후 자동 시작) ─────────────
echo ""
echo "[6/6] systemd 서비스 등록..."
cd "$RUNNER_DIR"
sudo ./svc.sh install
sudo ./svc.sh start

echo "  ✅ 서비스 등록 및 시작"
echo "  상태 확인: sudo systemctl status actions.runner.*"

# ── 완료 ───────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  ✅ 셋업 완료!"
echo ""
echo "  다음 단계:"
echo "  1. GitHub 레포 → Settings → Actions → Runners 에서"
echo "     'oracle-kr-trader' 가 Online 상태인지 확인"
echo ""
echo "  2. 환경 변수 설정:"
echo "     sudo -u $(whoami) tee /etc/environment << 'EOF'"
echo "     KIS_APP_KEY=your_key"
echo "     KIS_APP_SECRET=your_secret"
echo "     KIS_ACCOUNT=your_account"
echo "     KRX_ID=your_id"
echo "     KRX_PW=your_pw"
echo "     TELEGRAM_BOT_TOKEN=your_token"
echo "     TELEGRAM_CHAT_ID=your_chat_id"
echo "     EOF"
echo ""
echo "  3. 워크플로우 파일 업데이트 후 push"
echo "======================================================"
