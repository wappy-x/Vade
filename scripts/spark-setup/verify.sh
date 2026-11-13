#!/usr/bin/env bash
# Preview checks by default; --apply runs checks and the selected bounded traffic.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$SCRIPT_DIR/common.sh"
apply=0
if [[ ${!#} == --apply ]]; then apply=1; set -- "${@:1:$#-1}"; fi
usage() { printf 'Usage: %s CONFIG [--bw-server ROW|--bw-client ROW] [--apply]\n' "$0" >&2; exit 2; }
[[ $# == 1 || $# == 3 ]] || usage
config=$1; mode=${2:-}; row=${3:-}
[[ -z $mode || $mode == --bw-server || $mode == --bw-client ]] || usage
load_config "$config"
[[ -z $mode || $row =~ ^[1-9]$ ]] || usage
[[ -z $mode || $row -le ${#NETDEVS[@]} ]] || usage
peer_ssh_args() {
    ssh_args=(-o "HostName=$1" -o ProxyCommand=none -o ProxyJump=none -o "HostKeyAlias=$1"
        -o PermitLocalCommand=no -o ClearAllForwardings=yes -o ForwardAgent=no -o ForwardX11=no -o KnownHostsCommand=none
        -o BatchMode=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no
        -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=5 -o ConnectionAttempts=1 -b "$2" -n)
    [[ -z ${SSH_KEY_FILE:-} ]] || ssh_args+=(-i "$SSH_KEY_FILE" -o IdentitiesOnly=yes)
}
bandwidth_args() {
    local i=$1
    args=(-d "${RDMA_DEVS[i]}" -i 1 -x "${GID_INDEXES[i]}" -p "${BW_PORTS[i]}" --bind_source_ip="${LOCAL_CIDRS[i]%/*}" -m 4096 -F --report_gbits -D 5 -s 65536)
    [[ $mode != --bw-client ]] || args+=("${PEER_IPS[i]}")
}
gid_matches_ipv4() {
    local actual expected octet a b c d extra
    actual=$(cat -- "$1") || return 1
    # Linux sysfs emits eight full hextets. Accept only that documented form;
    # compressed IPv6 or extra text needs a structured parser and is refused.
    [[ $actual =~ ^0000:0000:0000:0000:0000:[fF]{4}:[0-9a-fA-F]{4}:[0-9a-fA-F]{4}$ ]] || return 1
    [[ $2 =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || return 1
    IFS=. read -r a b c d extra <<<"$2"
    [[ -z $extra ]] || return 1
    for octet in "$a" "$b" "$c" "$d"; do
        [[ $octet =~ ^(0|[1-9][0-9]{0,2})$ && $octet -le 255 ]] || return 1
    done
    expected=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' "$a" "$b" "$c" "$d") || return 1
    actual=$(printf '%s' "$actual" | tr A-F a-f) || return 1
    [[ $actual == "$expected" ]]
}
verbs_cache_present() {
    # Cache registration is diagnostic; ibv_devinfo below must actually open
    # every selected rail through libibverbs and its provider as this caller.
    ldconfig -p | awk '$1 == "libibverbs.so.1" { found=1 } END { exit !found }'
}
if [[ $apply == 0 ]]; then
    command_plan uname -s
    command_plan uname -m
    command_plan uname -r
    command_plan id
    command_plan ulimit -Sl
    command_plan ulimit -Hl
    command_plan ldconfig -p
    command_plan nvidia-smi --query-gpu=name --format=csv,noheader
    command_plan cat /proc/cmdline
    if [[ -z $mode ]]; then
        [[ -z $SERVICE_UNIT ]] || command_plan systemctl show "$SERVICE_UNIT" -p User -p LimitMEMLOCK -p LimitMEMLOCKSoft -p MainPID
        command_plan bash "$SCRIPT_DIR/updates.sh" status
    fi
    for i in "${!NETDEVS[@]}"; do
        [[ -z $mode || $((i+1)) == "$row" ]] || continue
        dev=${NETDEVS[i]}; port=/sys/class/infiniband/${RDMA_DEVS[i]}/ports/1; gid=${GID_INDEXES[i]}
        command_plan ip -4 -o address show dev "$dev"
        command_plan cat "/sys/class/net/$dev/mtu" "/sys/class/net/$dev/carrier" "/sys/class/net/$dev/speed"
        command_plan cat "$port/state" "$port/link_layer" "$port/gid_attrs/ndevs/$gid" "$port/gid_attrs/types/$gid" "$port/gids/$gid"
        command_plan ls -l /dev/infiniband
        command_plan timeout 8s ibv_devinfo -d "${RDMA_DEVS[i]}" -i 1
        if [[ -z $mode ]]; then
            command_plan getent ahostsv4 "${PEER_NAMES[i]}"
            command_plan ping -I "$dev" -M do -s 8972 -c 1 -W 2 "${PEER_IPS[i]}"
            peer_ssh_args "${PEER_IPS[i]}" "${LOCAL_CIDRS[i]%/*}"
            command_plan timeout 10s ssh "${ssh_args[@]}" "$SERVICE_USER@${PEER_IPS[i]}" 'printf "%s %s %s\n" "$(id -un)" "$(ulimit -Sl)" "$(ulimit -Hl)"'
        fi
    done
    if [[ -n $mode ]]; then bandwidth_args "$((row-1))"; command_plan timeout 25s ib_write_bw "${args[@]}"; fi
    echo 'PREVIEW checks only; no file diffs (verification writes no configuration); add --apply to run checks/traffic'
    exit
fi
failures=0; unverified=0
result() {
    printf '%s %s\n' "$1" "$2"
    case $1 in FAIL) failures=$((failures + 1));; UNVERIFIED) unverified=$((unverified + 1));; esac
}
check() { local label=$1; shift; if "$@" >/dev/null 2>&1; then result PASS "$label"; else result FAIL "$label"; fi; }
output_equals() {
    local expected=$1 actual; shift
    actual=$("$@") || return 1
    [[ $actual == "$expected" ]]
}
output_matches() {
    local pattern=$1 actual; shift
    actual=$("$@") || return 1
    [[ $actual != *$'\n'* && $actual =~ $pattern ]]
}
system=$(uname -s) || { result FAIL 'cannot inspect operating system'; exit 1; }
[[ $system == Linux ]] || { result FAIL 'requires DGX OS Linux'; exit 1; }
check 'aarch64 architecture' output_equals aarch64 uname -m
caller_uid=$(id -u) || { result FAIL 'cannot inspect caller identity'; exit 1; }
caller_name=$(id -un) || { result FAIL 'cannot inspect caller account'; exit 1; }
[[ $caller_uid =~ ^(0|[1-9][0-9]*)$ ]] || { result FAIL 'caller numeric identity is ambiguous'; exit 1; }
if [[ $caller_uid != 0 && $caller_name == "$SERVICE_USER" ]]; then
    result PASS 'running as the configured service user'
else result FAIL 'run from the service user launch session, without sudo'; fi
check 'actual launch-session soft memlock unlimited' output_equals unlimited ulimit -Sl
check 'actual launch-session hard memlock unlimited' output_equals unlimited ulimit -Hl
check 'libibverbs registered in system loader cache' verbs_cache_present
check 'GB10 driver available' bash -o pipefail -c 'nvidia-smi --query-gpu=name --format=csv,noheader | grep -q GB10'
kernel=$(uname -r) || { result FAIL 'cannot inspect running kernel'; exit 1; }
if [[ $kernel == 7.0.0-1019-nvidia ]]; then
    if awk '{ for (i=1; i<=NF; i++) if ($i ~ /^kho=/) { found=1; if ($i != "kho=off") bad=1 } } END { exit (!found || bad) }' /proc/cmdline; then
        result PASS 'known affected kernel has kho=off (memory-pressure behavior UNVERIFIED)'
    else result FAIL 'known affected kernel requires kho=off without conflicting kho options, then a reboot'; fi
elif [[ -n ${KERNEL_FIXED:-} && $kernel == "$KERNEL_FIXED" ]]; then
    result PASS 'running the exact operator-approved fixed kernel'
else result FAIL "kernel $kernel is UNVERIFIED; review advisory and approve an exact KERNEL_FIXED"; fi
if [[ -z $mode ]]; then
    if [[ -n ${SERVICE_UNIT:-} ]]; then
        check 'service user matches config' output_equals "$SERVICE_USER" systemctl show "$SERVICE_UNIT" -p User --value
        check 'service memlock unlimited' output_equals infinity systemctl show "$SERVICE_UNIT" -p LimitMEMLOCK --value
        check 'service soft memlock unlimited' output_equals infinity systemctl show "$SERVICE_UNIT" -p LimitMEMLOCKSoft --value
        if ! pid=$(systemctl show "$SERVICE_UNIT" -p MainPID --value 2>/dev/null); then
            result FAIL 'cannot inspect configured service process'
        elif [[ $pid =~ ^[1-9][0-9]*$ ]]; then
            check 'running service soft/hard memlock unlimited' awk '$1 == "Max" && $2 == "locked" && $3 == "memory" { found=1; if ($4 != "unlimited" || $5 != "unlimited") exit 1 } END { if (!found) exit 1 }' "/proc/$pid/limits"
        elif [[ $pid == 0 ]]; then result UNVERIFIED 'service is not running; verify its actual process limits when launched';
        else result FAIL 'configured service process identity is ambiguous'; fi
    fi
    for unit in apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service apt-daily.service apt-daily-upgrade.service; do
        if ! state=$(systemctl show "$unit" -p LoadState --value 2>/dev/null) ||
            ! active=$(systemctl show "$unit" -p ActiveState --value 2>/dev/null); then
            result FAIL "$unit cannot be inspected"; continue
        fi
        if [[ $state == not-found && $active == inactive ]]; then result PASS "$unit absent and inactive (nothing to pause)"; continue; fi
        if [[ $state != loaded && $state != masked ]]; then result FAIL "$unit load state is unsupported"; continue; fi
        if ! enabled=$(systemctl show "$unit" -p UnitFileState --value 2>/dev/null); then result FAIL "$unit enablement cannot be inspected"; continue; fi
        if [[ ( $enabled == masked || $enabled == masked-runtime ) && $active == inactive ]]; then result PASS "$unit paused";
        else result FAIL "$unit must be masked and inactive for benchmarking"; fi
    done
fi
for ((i=0; i<${#NETDEVS[@]}; i++)); do
    [[ -z $mode || $((i+1)) == "$row" ]] || continue
    netdev=${NETDEVS[i]}; rdma=${RDMA_DEVS[i]}; cidr=${LOCAL_CIDRS[i]}; gid=${GID_INDEXES[i]}
    local_ip=${cidr%/*}; peer=${PEER_IPS[i]}; name=${PEER_NAMES[i]}
    prefix="rail $((i+1))"
    check "$prefix configured address" bash -o pipefail -c 'ip -4 -o address show dev "$1" | awk "{print \$4}" | grep -Fxq -- "$2"' _ "$netdev" "$cidr"
    check "$prefix MTU 9000" output_equals 9000 cat "/sys/class/net/$netdev/mtu"
    check "$prefix carrier up" output_equals 1 cat "/sys/class/net/$netdev/carrier"
    speed=$(cat "/sys/class/net/$netdev/speed" 2>/dev/null) || speed=
    if [[ $speed =~ ^[0-9]+$ && $speed -ge 100000 ]]; then result PASS "$prefix link speed $speed Mb/s";
    else result FAIL "$prefix link speed must be at least 100000 Mb/s (observed ${speed:-unknown})"; fi
    port="/sys/class/infiniband/$rdma/ports/1"
    check "$prefix RDMA ACTIVE" output_matches '^4:[[:space:]]*ACTIVE$' cat "$port/state"
    check "$prefix RDMA Ethernet link layer" output_equals Ethernet cat "$port/link_layer"
    check "$prefix GID maps to netdev" output_equals "$netdev" cat "$port/gid_attrs/ndevs/$gid"
    check "$prefix RoCE v2 GID" output_equals 'RoCE v2' cat "$port/gid_attrs/types/$gid"
    check "$prefix canonical IPv4-mapped GID matches local IPv4" gid_matches_ipv4 "$port/gids/$gid" "$local_ip"
    access=0
    for verbs in "/sys/class/infiniband/$rdma/device/infiniband_verbs"/uverbs*; do
        [[ -e $verbs ]] || continue
        node="/dev/infiniband/${verbs##*/}"
        [[ -r $node && -w $node ]] && access=1
    done
    if [[ $access == 1 && $caller_uid != 0 ]]; then result PASS "$prefix uverbs accessible to caller";
    else result FAIL "$prefix no mapped uverbs node readable/writable by the non-root caller"; fi
    if info=$(timeout 8s ibv_devinfo -d "$rdma" -i 1 2>/dev/null); then
        result PASS "$prefix verbs provider opens device"
        check "$prefix RDMA active MTU 4096" grep -Eq 'active_mtu:[[:space:]]*4096([[:space:]]|$)' <<< "$info"
    else result FAIL "$prefix verbs provider query failed or timed out"; fi
    if [[ -z $mode ]]; then
        if resolved=$(getent ahostsv4 "$name" 2>/dev/null | awk '{print $1}' | sort -u); then
            check "$prefix peer name resolves uniquely to expected rail address" test "$resolved" = "$peer"
        else result FAIL "$prefix peer name resolution failed"; fi
        check "$prefix 9000-byte IPv4 frame reaches peer" ping -I "$netdev" -M do -s 8972 -c 1 -W 2 "$peer"
        peer_ssh_args "$peer" "$local_ip"
        if reply=$(timeout 10s ssh "${ssh_args[@]}" "$SERVICE_USER@$peer" 'printf "%s %s %s\n" "$(id -un)" "$(ulimit -Sl)" "$(ulimit -Hl)"' 2>/dev/null) && [[ $reply == "$SERVICE_USER unlimited unlimited" ]]; then
            result PASS "$prefix trusted noninteractive peer SSH and SSH-session memlock"
        else result FAIL "$prefix peer SSH or its actual session memlock"; fi
        result UNVERIFIED "$prefix ib_write_bw not run; explicitly test both directions before Vade"
    fi
done
if [[ -n $mode ]]; then
    i=$((row-1))
    if [[ $failures != 0 ]]; then result FAIL 'bandwidth not run because local preflight failed';
    elif ! command -v timeout >/dev/null || ! command -v ib_write_bw >/dev/null; then result FAIL 'bandwidth requires timeout and ib_write_bw';
    else
        bandwidth_args "$i"
        printf 'INFO rail %s %s: 5-second RDMA write; reported bandwidth units Gb/s; 25-second timeout\n' "$row" "$mode"
        if output=$(timeout 25s ib_write_bw "${args[@]}" 2>&1); then
            printf '%s\n' "$output"
            if printf '%s\n' "$output" | awk '$1 == 65536 && $2 ~ /^[0-9]+$/ && $4+0 > 0 { ok=1 } END { exit !ok }'; then
                result PASS "rail $row bounded ib_write_bw completed with positive client-reported average Gb/s"
            else result FAIL "rail $row ib_write_bw returned no positive 65536-byte result"; fi
        else printf '%s\n' "$output"; result FAIL "rail $row ib_write_bw failed or timed out"; fi
    fi
    printf 'INFO scope=one bandwidth row/direction; retain every rail and reverse-direction result\n'
else printf 'INFO scope=prerequisites; bandwidth and fresh-OS qualification remain UNVERIFIED\n'; fi
printf 'INFO failures=%s unverified=%s\n' "$failures" "$unverified"
[[ $failures == 0 ]]
