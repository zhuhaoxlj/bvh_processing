#!/usr/bin/env bash

set -uo pipefail

SCRIPT_NAME=${0##*/}
CACHE_DIR=${XDG_CACHE_HOME:-$HOME/.cache}/cursor-server-offline
COMMIT=""
JOBS=1
FORCE=0
DOWNLOAD_ONLY=0
CONNECT_TIMEOUT=15
REMOTE_BASE='.cursor-server'
DOWNLOADED_ARCHIVE=""
SSH_OPTIONS=()
HOSTS=()

usage() {
  cat <<'EOF'
Install Cursor Server on SSH hosts that cannot access the internet.

The archive is downloaded once on the local machine, cached, copied over SSH,
and installed atomically on each remote host.

Usage:
  install_cursor_server_offline.sh [options] HOST [HOST ...]

Options:
  --commit SHA          Cursor commit. Default: detect from `cursor --version`.
  --cache-dir PATH      Local archive cache directory.
  --jobs N              Number of hosts to install concurrently (default: 1).
  --connect-timeout N   SSH connection timeout in seconds (default: 15).
  --remote-base PATH    Directory below remote $HOME (default: .cursor-server).
  --ssh-option OPTION   Extra SSH/scp -o option; may be repeated.
  --force               Reinstall even when the requested commit is complete.
  --download-only       Download/cache packages after detecting host arches.
  -h, --help            Show this help.

Examples:
  ./scripts/install_cursor_server_offline.sh gpu robot01 robot02
  ./scripts/install_cursor_server_offline.sh --jobs 4 robot{01..12}
  ./scripts/install_cursor_server_offline.sh --commit 63a2996a10d9e476b6c28e951dd7691d9c0cf480 gpu

The remote host needs: Linux, ssh, tar, gzip, mkdir, mv, and standard POSIX tools.
Supported architectures: x86_64/amd64 (x64) and aarch64/arm64 (arm64).
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

log() {
  printf '[%s] %s\n' "$SCRIPT_NAME" "$*"
}

is_positive_integer() {
  [[ $1 =~ ^[1-9][0-9]*$ ]]
}

detect_cursor_commit() {
  local detected
  command -v cursor >/dev/null 2>&1 || return 1
  detected=$(cursor --version 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-fA-F]{40}$' | head -1)
  [[ $detected =~ ^[0-9a-fA-F]{40}$ ]] || return 1
  printf '%s\n' "${detected,,}"
}

normalize_arch() {
  case "$1" in
    x86_64 | amd64) printf 'x64\n' ;;
    aarch64 | arm64) printf 'arm64\n' ;;
    *) return 1 ;;
  esac
}

validate_archive() {
  local archive=$1
  local arch=$2
  [[ -s $archive ]] || return 1
  gzip -t "$archive" >/dev/null 2>&1 || return 1
  tar -tzf "$archive" "vscode-reh-linux-${arch}/node" >/dev/null 2>&1 || return 1
}

download_archive() {
  local arch=$1
  local cache_path="$CACHE_DIR/$COMMIT/linux-$arch/cursor-reh-linux-$arch.tar.gz"
  local temp_path="$cache_path.part.$$"
  local url="https://downloads.cursor.com/production/$COMMIT/linux/$arch/cursor-reh-linux-$arch.tar.gz"

  if validate_archive "$cache_path" "$arch"; then
    log "Using cached $arch package: $cache_path"
    DOWNLOADED_ARCHIVE=$cache_path
    return 0
  fi

  mkdir -p "${cache_path%/*}"
  if [[ -e $cache_path ]]; then
    mv "$cache_path" "$cache_path.invalid.$(date +%s)"
  fi

  log "Downloading Cursor Server $COMMIT ($arch)"
  if command -v curl >/dev/null 2>&1; then
    if ! curl -fL --retry 3 --connect-timeout 15 --output "$temp_path" "$url"; then
      [[ ! -e $temp_path ]] || unlink "$temp_path"
      return 1
    fi
  elif command -v wget >/dev/null 2>&1; then
    if ! wget --tries=3 --timeout=15 --output-document="$temp_path" "$url"; then
      [[ ! -e $temp_path ]] || unlink "$temp_path"
      return 1
    fi
  else
    die 'Neither curl nor wget is installed locally.'
  fi

  if ! validate_archive "$temp_path" "$arch"; then
    [[ ! -e $temp_path ]] || unlink "$temp_path"
    return 1
  fi
  mv "$temp_path" "$cache_path"
  log "Cached $arch package: $cache_path ($(stat -c %s "$cache_path") bytes)"
  DOWNLOADED_ARCHIVE=$cache_path
}

remote_arch() {
  local host=$1
  local raw_arch
  local os_name machine
  raw_arch=$(ssh "${SSH_OPTIONS[@]}" "$host" \
    'printf "__CURSOR_OS__%s\n__CURSOR_ARCH__%s\n" "$(uname -s)" "$(uname -m)"') || return 1
  os_name=$(sed -n 's/^__CURSOR_OS__//p' <<<"$raw_arch" | tail -1)
  machine=$(sed -n 's/^__CURSOR_ARCH__//p' <<<"$raw_arch" | tail -1)
  [[ $os_name == Linux ]] || {
    printf 'Unsupported OS on %s: %s\n' "$host" "$os_name" >&2
    return 1
  }
  normalize_arch "$machine" || {
    printf 'Unsupported architecture on %s: %s\n' "$host" "$machine" >&2
    return 1
  }
}

install_host() {
  local host=$1
  local arch=$2
  local archive=$3
  local remote_archive="/tmp/cursor-reh-${COMMIT}-${arch}-$$.tar.gz"

  printf '[%s] Checking existing installation\n' "$host"
  if [[ $FORCE -eq 0 ]] && ssh "${SSH_OPTIONS[@]}" "$host" bash -s -- "$COMMIT" "$arch" "$REMOTE_BASE" <<'REMOTE_CHECK'
set -eu
commit=$1
arch=$2
remote_base=$3
case "$remote_base" in
  /*) base=$remote_base ;;
  *) base=$HOME/$remote_base ;;
esac
target=$base/bin/linux-$arch/$commit
test -x "$target/node"
test -x "$target/bin/cursor-server"
version_output=$("$target/bin/cursor-server" --version 2>/dev/null)
printf '%s\n' "$version_output" | grep -Fq "$commit"
REMOTE_CHECK
  then
    printf '[%s] Already installed and verified; skipping\n' "$host"
    return 0
  fi

  printf '[%s] Uploading %s\n' "$host" "${archive##*/}"
  scp "${SSH_OPTIONS[@]}" "$archive" "$host:$remote_archive" || return 1

  printf '[%s] Installing Cursor Server %s (%s)\n' "$host" "$COMMIT" "$arch"
  ssh "${SSH_OPTIONS[@]}" "$host" bash -s -- \
    "$COMMIT" "$arch" "$REMOTE_BASE" "$remote_archive" "$FORCE" <<'REMOTE_INSTALL'
set -eu

commit=$1
arch=$2
remote_base=$3
archive=$4
force=$5

case "$remote_base" in
  /*) base=$remote_base ;;
  *) base=$HOME/$remote_base ;;
esac

platform_dir=$base/bin/linux-$arch
target=$platform_dir/$commit
stage=$platform_dir/.${commit}.install.$$
lock_dir=/tmp/cursor-offline-install-$(id -u)-${commit}-${arch}.lock

cleanup() {
  status=$?
  trap - EXIT HUP INT TERM
  if [ -e "$archive" ]; then unlink "$archive"; fi
  if [ -d "$stage" ]; then
    find "$stage" -depth -mindepth 1 -delete 2>/dev/null || true
    rmdir "$stage" 2>/dev/null || true
  fi
  if [ -d "$lock_dir" ]; then rmdir "$lock_dir" 2>/dev/null || true; fi
  exit "$status"
}
trap cleanup EXIT HUP INT TERM

mkdir -p "$platform_dir"

attempt=0
while ! mkdir "$lock_dir" 2>/dev/null; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 60 ]; then
    echo "Could not acquire offline install lock: $lock_dir" >&2
    exit 1
  fi
  sleep 1
done

if [ "$force" -eq 0 ] && [ -x "$target/node" ] && [ -x "$target/bin/cursor-server" ]; then
  version_output=$("$target/bin/cursor-server" --version 2>/dev/null || true)
  if printf '%s\n' "$version_output" | grep -Fq "$commit"; then
    echo "Another process completed the installation; skipping"
    exit 0
  fi
fi

mkdir "$stage"
tar -xzf "$archive" --strip-components=1 -C "$stage"
chmod +x "$stage/node" "$stage/bin/cursor-server"
test -x "$stage/node"
test -x "$stage/bin/cursor-server"

version_output=$("$stage/bin/cursor-server" --version 2>/dev/null)
printf '%s\n' "$version_output" | grep -Fq "$commit"

if [ -e "$target" ]; then
  backup=$platform_dir/${commit}.incomplete.$(date +%Y%m%d-%H%M%S).$$
  mv "$target" "$backup"
  echo "Previous installation preserved at: $backup"
fi
mv "$stage" "$target"

"$target/node" --version
"$target/bin/cursor-server" --version
REMOTE_INSTALL
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit)
      [[ $# -ge 2 ]] || die '--commit requires a value.'
      COMMIT=$2
      shift 2
      ;;
    --cache-dir)
      [[ $# -ge 2 ]] || die '--cache-dir requires a value.'
      CACHE_DIR=$2
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || die '--jobs requires a value.'
      JOBS=$2
      shift 2
      ;;
    --connect-timeout)
      [[ $# -ge 2 ]] || die '--connect-timeout requires a value.'
      CONNECT_TIMEOUT=$2
      shift 2
      ;;
    --remote-base)
      [[ $# -ge 2 ]] || die '--remote-base requires a value.'
      REMOTE_BASE=$2
      shift 2
      ;;
    --ssh-option)
      [[ $# -ge 2 ]] || die '--ssh-option requires a value.'
      SSH_OPTIONS+=(-o "$2")
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --download-only)
      DOWNLOAD_ONLY=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      HOSTS+=("$@")
      break
      ;;
    -*) die "Unknown option: $1" ;;
    *)
      HOSTS+=("$1")
      shift
      ;;
  esac
done

is_positive_integer "$JOBS" || die '--jobs must be a positive integer.'
is_positive_integer "$CONNECT_TIMEOUT" || die '--connect-timeout must be a positive integer.'
[[ ${#HOSTS[@]} -gt 0 ]] || die 'Provide at least one SSH host.'
case "$REMOTE_BASE" in
  '' | / | . | ..) die '--remote-base must name a dedicated Cursor Server directory.' ;;
esac

if [[ -z $COMMIT ]]; then
  COMMIT=$(detect_cursor_commit) || die 'Could not detect Cursor commit. Pass --commit SHA.'
fi
[[ $COMMIT =~ ^[0-9a-fA-F]{40}$ ]] || die '--commit must be a 40-character hexadecimal SHA.'
COMMIT=${COMMIT,,}

SSH_OPTIONS=(-o "ConnectTimeout=$CONNECT_TIMEOUT" "${SSH_OPTIONS[@]}")

declare -A HOST_ARCH=()
declare -A ARCHIVE_PATH=()
declare -A SEEN_ARCH=()
detect_failures=0

log "Cursor commit: $COMMIT"
for host in "${HOSTS[@]}"; do
  log "Detecting $host"
  if arch=$(remote_arch "$host"); then
    HOST_ARCH["$host"]=$arch
    SEEN_ARCH["$arch"]=1
    log "$host: linux-$arch"
  else
    printf '[%s] Architecture detection failed\n' "$host" >&2
    detect_failures=$((detect_failures + 1))
  fi
done

for arch in "${!SEEN_ARCH[@]}"; do
  download_archive "$arch" || die "Failed to download/validate $arch package."
  ARCHIVE_PATH["$arch"]=$DOWNLOADED_ARCHIVE
done

if [[ $DOWNLOAD_ONLY -eq 1 ]]; then
  log 'Download-only operation complete.'
  [[ $detect_failures -eq 0 ]]
  exit
fi

temp_logs=$(mktemp -d)
cleanup_local() {
  find "$temp_logs" -type f -delete 2>/dev/null || true
  rmdir "$temp_logs" 2>/dev/null || true
}
trap cleanup_local EXIT HUP INT TERM

pids=()
declare -A PID_HOST=()
declare -A PID_LOG=()
install_failures=$detect_failures

reap_first() {
  local pid=${pids[0]}
  local host=${PID_HOST[$pid]}
  local output=${PID_LOG[$pid]}
  if wait "$pid"; then
    cat "$output"
    printf '[%s] PASS\n' "$host"
  else
    cat "$output" >&2
    printf '[%s] FAIL\n' "$host" >&2
    install_failures=$((install_failures + 1))
  fi
  pids=("${pids[@]:1}")
}

for host in "${HOSTS[@]}"; do
  [[ -n ${HOST_ARCH[$host]+x} ]] || continue
  arch=${HOST_ARCH[$host]}
  output=$temp_logs/${#pids[@]}-${host//[^a-zA-Z0-9_.-]/_}.log
  install_host "$host" "$arch" "${ARCHIVE_PATH[$arch]}" >"$output" 2>&1 &
  pid=$!
  pids+=("$pid")
  PID_HOST["$pid"]=$host
  PID_LOG["$pid"]=$output
  if [[ ${#pids[@]} -ge $JOBS ]]; then
    reap_first
  fi
done

while [[ ${#pids[@]} -gt 0 ]]; do
  reap_first
done

if [[ $install_failures -gt 0 ]]; then
  die "$install_failures host(s) failed."
fi

log "All ${#HOST_ARCH[@]} host(s) are ready."
