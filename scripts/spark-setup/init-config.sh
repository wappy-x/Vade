#!/usr/bin/env bash
# Create placeholders in ignored local storage; never overwrite an operator's file.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
apply=0
if [[ ${!#} == --apply ]]; then apply=1; set -- "${@:1:$#-1}"; fi
[[ $# == 2 && ( $1 == pair || $1 == ring ) ]] || { echo 'usage: init-config.sh pair|ring .local/spark-setup/NODE [--apply]' >&2; exit 2; }
umask 077
git check-ignore --quiet -- "$2/config.env" || { echo 'FAIL use an ignored .local/ directory in this checkout' >&2; exit 1; }
[[ ! -e $2/config.env && ! -e $2/links.tsv ]] || { echo 'FAIL config already exists' >&2; exit 1; }
config=$(cat <<'EOF'
# Trusted shell code. Fill on this node; never stage this directory.
SERVICE_USER=REPLACE_SERVICE_USER
RDMA_GROUP=vade-rdma
LINKS_FILE=links.tsv
NETPLAN_RENDERER=NetworkManager
SSH_KEY_FILE="$HOME/.ssh/vade-cluster"
# Optional existing Vade systemd unit; empty for INSTALL.md's SSH launch.
SERVICE_UNIT=
# Exact running kernel approved from NVIDIA release evidence and RDMA checks.
# Leave empty for 7.0.0-1019-nvidia + kho=off. No fixed build is yet verified.
KERNEL_FIXED=
EOF
)
links=$(
printf '# netdev rdma_device local_ipv4/24 peer_ipv4 peer_alias ipv4_rocev2_gid test_tcp_port\n'
printf 'REPLACE_NETDEV_A0 REPLACE_RDMA_A0 REPLACE_LOCAL_A0 REPLACE_PEER_A0 REPLACE_ALIAS_A0 REPLACE_GID_A0 18515\n'
printf 'REPLACE_NETDEV_A1 REPLACE_RDMA_A1 REPLACE_LOCAL_A1 REPLACE_PEER_A1 REPLACE_ALIAS_A1 REPLACE_GID_A1 18516\n'
if [[ $1 == ring ]]; then
    printf 'REPLACE_NETDEV_B0 REPLACE_RDMA_B0 REPLACE_LOCAL_B0 REPLACE_PEER_B0 REPLACE_ALIAS_B0 REPLACE_GID_B0 18517\n'
    printf 'REPLACE_NETDEV_B1 REPLACE_RDMA_B1 REPLACE_LOCAL_B1 REPLACE_PEER_B1 REPLACE_ALIAS_B1 REPLACE_GID_B1 18518\n'
fi

)
check_path "$2/config.env"
check_path "$2/links.tsv"
command_plan mkdir -p -- "$2"
file_diff "$2/config.env" "$config"
file_diff "$2/links.tsv" "$links"
file_write_plan "$2/config.env" "$config"
file_write_plan "$2/links.tsv" "$links"
if [[ $apply == 0 ]]; then echo 'PREVIEW only; add --apply to create placeholders'; exit; fi
mkdir -p -- "$2"
printf '%s\n' "$config" > "$2/config.env"
printf '%s\n' "$links" > "$2/links.tsv"
chmod 600 "$2/config.env" "$2/links.tsv"
echo 'PASS local placeholders created; replace every REPLACE field before rendering'
