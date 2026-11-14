#!/usr/bin/env bash
# Read-only validation of one private setup recovery receipt; never source it.
set -euo pipefail
fail() { printf 'FAIL recovery: %s\n' "$*" >&2; exit 1; }
[[ $# == 3 ]] || fail 'usage: transaction-check.sh SNAPSHOT SERVICE_USER RDMA_GROUP (read-only)'
TX=$1; user=$2; group=$3
[[ $user =~ ^[a-z_][a-z0-9_-]*$ && $group =~ ^[a-z_][a-z0-9_-]*$ ]] || fail 'invalid account/group identity'
ancestors() {
    local path=$1
    while :; do
        [[ ! -L $path ]] || fail 'destination contains a symlink'
        [[ $path != / ]] || break
        path=$(dirname "$path")
        [[ ! -e $path || -d $path ]] || fail 'destination ancestor is not a directory'
    done
}
receipt_file() {
    local owner
    ancestors "$1"
    [[ -f $1 && ! -L $1 ]] || fail 'unsafe receipt file'
    owner=$(stat -c %u "$1") || fail 'cannot inspect receipt ownership'
    [[ $owner == 0 ]] || fail 'unsafe receipt file'
    [[ -r $1 ]] && cat -- "$1" >/dev/null || fail 'cannot read receipt file'
}
safe_target() {
    local owner
    ancestors "$1"
    [[ ! -e $1 || -f $1 ]] || fail 'destination is not a regular file'
    if [[ -e $1 ]]; then
        owner=$(stat -c %u "$1") || fail 'cannot inspect destination ownership'
        [[ $owner == 0 ]] || fail 'destination is no longer root-owned'
    fi
}
contains() {
    local expected=$1 value; shift
    for value in "$@"; do [[ $value != "$expected" ]] || return 0; done
    return 1
}
same_bytes() {
    local status=0
    cmp -s "$1" "$2" || status=$?
    [[ $status -le 1 ]] || fail 'cannot compare destination and receipt bytes'
    return "$status"
}
complete_lines() {
    # Restore consumers use complete TSV/text lines. Refuse truncated receipts
    # rather than validate a final record that those loops would not restore.
    # Bash variables cannot retain NUL; never let read normalize a malformed
    # path, identity or permission field into an allowed value.
    LC_ALL=C tr -d '\000' < "$1" | cmp -s "$1" - || fail 'NUL or unreadable data in line-oriented recovery receipt'
    [[ ! -s $1 ]] || tail -c 1 -- "$1" | cmp -s - <(printf '\n') || fail 'unterminated line in recovery receipt'
}
children() {
    # Unmatched globs are skipped by callers; this also covers hidden entries.
    [[ -r $1 && -x $1 ]] && ls -A -- "$1" >/dev/null || fail 'cannot enumerate directory contents'
    CHILDREN=("$1"/* "$1"/.[!.]* "$1"/..?*)
}
ancestors "$TX"
[[ -d $TX ]] || fail 'snapshot must be a directory'
metadata=$(stat -c '%u:%a' "$TX") || fail 'cannot inspect snapshot metadata'
[[ $metadata == 0:700 ]] || fail 'snapshot must be root-owned, mode 700'
for name in action identity.tsv ready files.tsv directories.txt; do receipt_file "$TX/$name"; done
for name in action identity.tsv files.tsv directories.txt; do complete_lines "$TX/$name"; done
awk 'NR!=1 || $0 !~ /^(network|rdma|kernel)$/ { bad=1 } END { exit bad || NR!=1 }' "$TX/action" || fail 'unsupported transaction'
action=$(cat "$TX/action")
awk -F '\t' -v user="$user" -v group="$group" '
    NR!=1 || NF!=4 || $1!=user || $2!=group || $3 !~ /^[0-9]+$/ || $4 !~ /^[0-9]+$/ { bad=1 }
    END { exit bad || NR!=1 }
' "$TX/identity.tsv" || fail 'configuration account/group differ from receipt'
IFS=$'\t' read -r saved_user saved_group saved_uid saved_gid < "$TX/identity.tsv"
account=$(getent passwd "$user") || fail 'cannot inspect service account'
awk -F: -v user="$user" -v uid="$saved_uid" -v gid="$saved_gid" '
    NR!=1 || NF!=7 || $1!=user || "x"$3!="x"uid || "x"$4!="x"gid { bad=1 }
    END { exit bad || NR!=1 }
' <<< "$account" || fail 'service account numeric identity changed since transaction'
awk -F '\t' '
    NF!=3 || "x"$1!="x"sprintf("%d", NR-1) || $3 !~ /^(present|absent)$/ { bad=1 }
    END { exit bad || NR==0 }
' "$TX/files.tsv" || fail 'invalid file manifest'
TARGETS=(); PRIORS=(); TEMPORARIES=(); ALLOWED_FILES=(action identity.tsv ready files.tsv directories.txt complete)
while IFS=$'\t' read -r index destination prior || [[ -n $index ]]; do
    contains "$destination" "${TARGETS[@]-}" && fail 'duplicate file manifest'
    TARGETS+=("$destination"); PRIORS+=("$prior")
    if [[ $prior == present ]]; then ALLOWED_FILES+=("prior-$index"); fi
    ALLOWED_FILES+=("desired-$index")
done < "$TX/files.tsv"
case $action in
network)
    [[ ${#TARGETS[@]} == 2 && ${TARGETS[0]} == /etc/netplan/40-cx7.yaml && ${TARGETS[1]} == /etc/hosts ]] || fail 'invalid network file allowlist' ;;
kernel)
    [[ ${#TARGETS[@]} == 2 && ${TARGETS[0]} == /etc/default/grub.d/zzz-vade-kho.cfg && ${TARGETS[1]} == /boot/grub/grub.cfg ]] || fail 'invalid kernel file allowlist'
    for index in "${!TARGETS[@]}"; do
        if [[ ${TARGETS[index]} == /boot/grub/grub.cfg ]]; then
            [[ ${PRIORS[index]} == present ]] || fail 'missing saved grub.cfg'
            ALLOWED_FILES+=("candidate-$index")
        fi
    done ;;
rdma)
    [[ ${#TARGETS[@]} == 3 || ${#TARGETS[@]} == 4 ]] || fail 'invalid RDMA file allowlist'
    for name in /etc/security/limits.d/99-vade-memlock.conf /etc/udev/rules.d/99-vade-rdma.rules /etc/systemd/system/ssh.service.d/99-vade-memlock.conf; do
        contains "$name" "${TARGETS[@]-}" || fail 'invalid RDMA file allowlist'
    done
    for name in "${TARGETS[@]-}"; do
        case $name in
        /etc/security/limits.d/99-vade-memlock.conf|/etc/udev/rules.d/99-vade-rdma.rules|/etc/systemd/system/ssh.service.d/99-vade-memlock.conf) ;;
        *) [[ $name =~ ^/etc/systemd/system/[a-zA-Z0-9_.@-]+\.service\.d/99-vade-memlock\.conf$ ]] || fail 'invalid optional service destination' ;;
        esac
    done
    ALLOWED_FILES+=(group.tsv group-created-gid devices.tsv) ;;
esac
for index in "${!TARGETS[@]}"; do
    destination=${TARGETS[index]}; prior=${PRIORS[index]}
    safe_target "$destination"
    if [[ $action == kernel || ( $action == network && $destination == /etc/netplan/40-cx7.yaml ) ]]; then
        temporary="${destination%/*}/.${destination##*/}.vade-${TX##*/}.tmp"
        TEMPORARIES+=("$temporary")
        ancestors "$temporary"
        if [[ -e $temporary ]]; then
            metadata=$(stat -c '%u:%a' "$temporary") || fail 'cannot inspect publication temporary metadata'
            [[ $metadata == 0:* ]] || fail 'unsafe publication temporary'
            if [[ -d $temporary ]]; then
                [[ $metadata == 0:700 ]] || fail 'unsafe publication temporary directory'
                children "$temporary"
                for child in "${CHILDREN[@]}"; do
                    [[ -e $child || -L $child ]] || continue
                    [[ $child == "$temporary/content" ]] || fail 'publication temporary has unrelated contents'
                    receipt_file "$child"
                done
            else [[ -f $temporary ]] || fail 'unsafe publication temporary'; fi
        fi
    fi
    desired=$TX/desired-$index; before=$TX/prior-$index; matches=0
    if [[ $prior == present ]]; then
        receipt_file "$before"
        if [[ -f $destination ]] && same_bytes "$destination" "$before"; then matches=1; fi
    fi
    if [[ -e $desired || -L $desired ]]; then
        receipt_file "$desired"
        if [[ -f $destination ]] && same_bytes "$destination" "$desired"; then matches=1; fi
    elif [[ $destination != /boot/grub/grub.cfg ]]; then fail 'missing installed file receipt'; fi
    # A killed generation has no recorded desired grub.cfg; only prior bytes
    # qualify until a validated candidate was recorded and synced for publication.
    if [[ -e $destination ]]; then
        [[ $matches == 1 ]] || fail 'a destination changed since this transaction; review before restoring'
    elif [[ $prior == present ]]; then fail 'a prior existing destination disappeared; review before restoring'; fi
done
DIRECTORIES=(); PARENTS=()
for destination in "${TARGETS[@]-}"; do
    parent=$(dirname "$destination")
    while [[ $parent != / ]]; do
        case $parent in /etc|/boot) ;; *) PARENTS+=("$parent") ;; esac
        parent=$(dirname "$parent")
    done
done
while IFS= read -r name || [[ -n $name ]]; do
    contains "$name" "${DIRECTORIES[@]-}" && fail 'duplicate directory receipt'
    contains "$name" "${PARENTS[@]-}" || fail 'directory outside transaction destinations'
    DIRECTORIES+=("$name")
    ancestors "$name"
    if [[ -e $name ]]; then
        [[ -d $name ]] || fail 'created directory has changed type'
        owner=$(stat -c %u "$name") || fail 'cannot inspect created directory ownership'
        [[ $owner == 0 ]] || fail 'created directory has changed owner'
    fi
done < "$TX/directories.txt"
for name in "${DIRECTORIES[@]-}"; do
    [[ -d $name ]] || continue
    children "$name"
    for child in "${CHILDREN[@]}"; do
        [[ -e $child || -L $child ]] || continue
        contains "$child" "${TARGETS[@]-}" "${TEMPORARIES[@]-}" "${DIRECTORIES[@]-}" || fail 'created directory has unrelated contents'
    done
done
if [[ $action == rdma ]]; then
    receipt_file "$TX/group.tsv"; receipt_file "$TX/devices.tsv"
    complete_lines "$TX/group.tsv"; complete_lines "$TX/devices.tsv"
    awk -F '\t' '
        NR!=1 || NF!=3 || $1 !~ /^(yes|no)$/ || $2 !~ /^(yes|no)$/ || ($1=="yes" && $3 !~ /^[0-9]+$/) || ($1=="no" && $3!="-") { bad=1 }
        END { exit bad || NR!=1 }
    ' "$TX/group.tsv" || fail 'invalid saved group identity'
    IFS=$'\t' read -r existed member prior_gid < "$TX/group.tsv"
    memberships=$(id -nG "$user") || fail 'cannot inspect service account membership'
    if [[ $member == yes ]]; then tr ' ' '\n' <<< "$memberships" | grep -Fxq "$group" || fail 'prior group membership disappeared after transaction'; fi
    if current=$(getent group "$group"); then
        awk -F: -v group="$group" 'NR!=1 || NF!=4 || $1!=group || $3 !~ /^[0-9]+$/ { bad=1 } END { exit bad || NR!=1 }' <<< "$current" || fail 'ambiguous current group'
        IFS=: read -r name password current_gid current_members <<< "$current"
        if [[ $existed == yes ]]; then expected=$prior_gid
        else
            receipt_file "$TX/group-created-gid"
            complete_lines "$TX/group-created-gid"
            awk 'NR!=1 || $0 !~ /^[0-9]+$/ { bad=1 } END { exit bad || NR!=1 }' "$TX/group-created-gid" || fail 'invalid created group identity'
            expected=$(cat "$TX/group-created-gid")
        fi
        [[ $current_gid == "$expected" ]] || fail 'group identity changed since transaction'
        if [[ $existed == no ]]; then
            IFS=, read -r -a members <<< "$current_members"
            for name in "${members[@]-}"; do [[ -z $name || $name == "$user" ]] || fail 'created group now has other members'; done
            accounts=$(getent passwd) || fail 'cannot inspect primary groups'
            awk -F: -v gid="$expected" 'NF!=7 || $4 !~ /^[0-9]+$/ || "x"$4=="x"gid { bad=1 } END { exit bad || NR==0 }' <<< "$accounts" || fail 'created group is now a primary group or account data is ambiguous'
        fi
    else
        status=$?
        [[ $status == 2 && -z $current ]] || fail 'cannot inspect current group'
        [[ $existed == no ]] || fail 'prior group disappeared'
    fi
    awk -F '\t' 'NF!=4 || seen[$1]++ { bad=1 } END { exit bad }' "$TX/devices.tsv" || fail 'invalid device receipt'
    while IFS=$'\t' read -r node uid gid mode || [[ -n $node ]]; do
        [[ $node =~ ^/dev/infiniband/(uverbs[0-9]+|rdma_cm)$ ]] || fail 'device outside RDMA allowlist'
        [[ $uid =~ ^[0-9]+$ && $gid =~ ^[0-9]+$ && $mode =~ ^[0-7]{3,4}$ ]] || fail 'invalid device permissions'
        ancestors "$node"
        [[ -c $node && ! -L $node ]] || fail 'RDMA device identity unavailable'
    done < "$TX/devices.tsv"
fi
children "$TX"
for path in "${CHILDREN[@]}"; do
    [[ -e $path || -L $path ]] || continue
    contains "$(basename "$path")" "${ALLOWED_FILES[@]-}" || fail 'unrecognized file in recovery receipt'
    receipt_file "$path"
done
printf 'PASS complete recovery preflight; prior state and current destinations agree\n'
