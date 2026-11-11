#!/usr/bin/env bash
# Trusted local configuration; no fleet defaults and no secret-file reads.
die() { printf 'FAIL %s\n' "$*" >&2; exit 1; }
require_root() {
    [[ $(uname -s) == Linux && $EUID == 0 ]] || die 'run this step as root on the Spark'
}
load_config() {
    [[ $# == 1 && -f $1 ]] || die 'provide a trusted local config.env'
    CONFIG_DIR=$(cd "$(dirname "$1")" && pwd)
    SERVICE_UNIT= KERNEL_FIXED= SSH_KEY_FILE= RDMA_GROUP=vade-rdma
    NETPLAN_RENDERER=NetworkManager
    # shellcheck disable=SC1090
    source "$CONFIG_DIR/$(basename "$1")" || die 'could not load configuration'
    [[ ${SERVICE_USER:-} =~ ^[a-z_][a-z0-9_-]*$ ]] || die 'set SERVICE_USER'
    [[ $RDMA_GROUP =~ ^[a-z_][a-z0-9_-]*$ ]] || die 'invalid RDMA_GROUP'
    [[ -z $SERVICE_UNIT || $SERVICE_UNIT =~ ^[a-zA-Z0-9_.@-]+\.service$ ]] || die 'invalid SERVICE_UNIT'
    [[ -z $KERNEL_FIXED || $KERNEL_FIXED =~ ^[a-zA-Z0-9_.+-]+$ ]] || die 'invalid KERNEL_FIXED'
    [[ $NETPLAN_RENDERER == NetworkManager ]] || die 'this recipe uses per-interface NetworkManager'
    [[ -n ${LINKS_FILE:-} ]] || die 'set LINKS_FILE'
    [[ $LINKS_FILE == /* ]] || LINKS_FILE=$CONFIG_DIR/$LINKS_FILE
    [[ -f $LINKS_FILE ]] || die 'LINKS_FILE missing'
    # Validate before emitting YAML, hosts entries or commands. Each rail gets
    # a distinct /24; peer aliases and TCP test ports must be unique locally.
    LC_ALL=C awk '
    function fail(message) { print "FAIL links row " NR ": " message > "/dev/stderr"; bad=1; exit 1 }
    function ipv4(value, parts, n, i) {
        n=split(value, parts, "."); if (n != 4) return 0
        for (i=1; i<=4; i++) if (parts[i] !~ /^(0|[1-9][0-9]*)$/ || length(parts[i])>3 || parts[i]+0>255) return 0
        return 1
    }
    /^[[:space:]]*$/ || /^#/ { next }
    {
        if (NF != 7) fail("expected seven fields")
        if ($1 !~ /^[A-Za-z_][A-Za-z0-9_.-]*$/ || $1 ~ /^(null|Null|NULL|true|True|TRUE|false|False|FALSE|yes|Yes|YES|no|No|NO|on|On|ON|off|Off|OFF|y|Y|n|N)$/ || $2 !~ /^[A-Za-z0-9_.-]+$/ || $5 !~ /^[A-Za-z0-9_.-]+$/) fail("invalid device/alias or ambiguous plain YAML Ethernet name")
        if (split($3, cidr, "/") != 2 || cidr[2] !~ /^24$/ || !ipv4(cidr[1], local_ip) || !ipv4($4, peer_ip)) fail("use canonical IPv4 endpoints of one /24")
        subnet=local_ip[1] "." local_ip[2] "." local_ip[3]
        if (subnet != peer_ip[1] "." peer_ip[2] "." peer_ip[3] || cidr[1] == $4) fail("use different endpoints of one /24")
        if (local_ip[4]+0 == 0 || local_ip[4]+0 == 255 || peer_ip[4]+0 == 0 || peer_ip[4]+0 == 255) fail("network/broadcast endpoint")
        if ($6 !~ /^[0-9]+$/ || length($6)>3 || $6+0>255 || $7 !~ /^[0-9]+$/ || length($7)>5 || $7+0<1024 || $7+0>65535) fail("invalid GID index or test port")
        if (netdev[$1]++ || rdma[$2]++ || networks[subnet]++ || aliases[$5]++ || ports[$7+0]++) fail("duplicate device, subnet, alias or test port")
        rows++
    }
    END { if (!bad && rows != 2 && rows != 4) { print "FAIL links: pair needs two rows; ring needs four" > "/dev/stderr"; exit 1 } }
    ' "$LINKS_FILE" || die 'invalid rail configuration'
    NETDEVS=(); RDMA_DEVS=(); LOCAL_CIDRS=(); PEER_IPS=()
    PEER_NAMES=(); GID_INDEXES=(); BW_PORTS=()
    while read -r netdev rdma cidr peer name gid port || [[ -n $netdev ]]; do
        [[ -n $netdev && $netdev != \#* ]] || continue
        NETDEVS+=("$netdev"); RDMA_DEVS+=("$rdma"); LOCAL_CIDRS+=("$cidr")
        PEER_IPS+=("$peer"); PEER_NAMES+=("$name"); GID_INDEXES+=("$gid"); BW_PORTS+=("$port")
    done < "$LINKS_FILE"
}

# A preview must not create directories, lock files or temporary files.
command_plan() { printf 'COMMAND'; printf ' %q' "$@"; printf '\n'; }
check_path() {
    local path=$1 parent
    [[ $path == /* || $path != -* ]] || die 'invalid destination'
    [[ ! -L $path ]] || die "refusing symlink: $path"
    [[ ! -e $path || -f $path ]] || die "not a regular file: $path"
    parent=$(dirname "$path")
    while [[ $parent != / && $parent != . ]]; do
        [[ ! -L $parent ]] || die "refusing symlink directory: $parent"
        [[ ! -e $parent || -d $parent ]] || die "not a directory: $parent"
        parent=$(dirname "$parent")
    done
}
file_diff() {
    local path=$1 content=$2 previous=/dev/null status=0
    check_path "$path"
    [[ ! -f $path ]] || previous=$path
    diff -u --label "$path (before)" --label "$path (proposed)" "$previous" <(printf '%s\n' "$content") || status=$?
    [[ $status -le 1 ]] || die "cannot compare $path"
}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/content.sh"
file_write_plan() {
    printf 'COMMAND printf %q %q > %q\n' '%s\n' "$2" "$1"
    command_plan chmod 600 "$1"
}
