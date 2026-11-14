#!/usr/bin/env bash
# Review the complete plan first; system changes require an explicit --apply.
set -eEuo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
apply=0
if [[ ${!#} == --apply ]]; then apply=1; set -- "${@:1:$#-1}"; fi
RESULT_FILE=
if [[ $# -ge 4 && ${@: -2:1} == --result-file ]]; then
    RESULT_FILE=${!#}; set -- "${@:1:$#-2}"
    [[ $apply == 1 ]] || die '--result-file is only for an explicit --apply'
fi
[[ $# == 2 || ( $# == 3 && $2 == rollback ) ]] || die 'usage: apply.sh CONFIG packages|network|rdma|kernel OR CONFIG rollback SNAPSHOT; optional --result-file FILE before --apply'
config=$1; action=$2
[[ $action =~ ^(packages|network|rdma|kernel|rollback)$ ]] || die 'unknown setup step'
load_config "$config"
require_root
umask 077
STATE=/var/lib/vade-spark-setup
TX=
getent passwd "$SERVICE_USER" >/dev/null || die 'create the configured service user first'
need() { local cmd; for cmd in "$@"; do command -v "$cmd" >/dev/null || die "missing prerequisite: $cmd"; done; }
need diff cmp stat install flock cp id awk sort sync mv ln chmod chown touch mkdir rm rmdir mktemp ls
if [[ -n $RESULT_FILE ]]; then
    check_path "$RESULT_FILE"
    [[ ! -e $RESULT_FILE && -d $(dirname "$RESULT_FILE") ]] || die 'result file must be absent in an existing private directory'
fi
# A separate, fresh result prevents a successful read-only no-op from being
# mistaken for staging that needs activation. Failed operations publish nothing.
result_report() {
    [[ -n $RESULT_FILE ]] || return 0
    local temporary
    temporary=$(mktemp "$(dirname "$RESULT_FILE")/.vade-result.XXXXXX") || return 1
    if printf '%s\n' "$1" > "$temporary" && chmod 644 "$temporary" &&
        sync -f "$temporary" && ln -T -- "$temporary" "$RESULT_FILE" &&
        sync -f "$(dirname "$RESULT_FILE")" && rm -- "$temporary"; then
        return 0
    else
        # Remove only our own newly linked marker if its durability check failed.
        # A concurrent foreign result stays intact.
        if [[ ! -L $RESULT_FILE && -f $RESULT_FILE && $RESULT_FILE -ef $temporary ]]; then
            rm -- "$RESULT_FILE" || { printf 'FAIL could not retract failed result publication\n' >&2; return 1; }
            sync -f "$(dirname "$RESULT_FILE")" || true
        fi
        rm -- "$temporary" || return 1
        printf 'FAIL result publication failed; an existing result was never overwritten\n' >&2
        return 1
    fi
}
# Check state and destination ancestry without creating a directory or lock.
check_state() {
    local path metadata
    for path in "$STATE" "$STATE/transactions"; do
        [[ ! -L $path ]] || die 'state directory must not be a symlink'
        if [[ -e $path ]]; then
            metadata=$(stat -c '%u:%a' "$path") || die 'cannot inspect state directory metadata'
            [[ -d $path && $metadata == 0:700 ]] || die 'state directories must be root-owned, mode 700'
        fi
    done
    check_path "$STATE/apply.lock"
    if [[ -e $STATE/apply.lock ]]; then
        metadata=$(stat -c %u "$STATE/apply.lock") || die 'cannot inspect lock ownership'
        [[ $metadata == 0 ]] || die 'lock must be root-owned'
    fi
}
check_state
TARGETS=(); CONTENTS=(); MODES=()
add_file() {
    local existing metadata
    check_path "$1"
    if [[ -e $1 ]]; then
        metadata=$(stat -c %u "$1") || die 'cannot inspect existing system file ownership'
        [[ $metadata == 0 ]] || die 'existing system files must be root-owned'
    fi
    for existing in "${!TARGETS[@]}"; do [[ ${TARGETS[existing]} != "$1" ]] || die 'duplicate setup destination'; done
    TARGETS+=("$1"); CONTENTS+=("$2"); MODES+=("$3")
}
files_identical() {
    local i
    for i in "${!TARGETS[@]}"; do
        [[ -f ${TARGETS[i]} ]] && cmp -s "${TARGETS[i]}" <(printf '%s\n' "${CONTENTS[i]}") || return 1
    done
}
plan_files() {
    local i
    for i in "${!TARGETS[@]}"; do
        file_diff "${TARGETS[i]}" "${CONTENTS[i]}"
        if [[ $action == network && $i == 0 ]]; then
            command_plan 'atomically publish complete netplan only if absent; leave identical bytes untouched' "${TARGETS[i]}"
        elif [[ $action == kernel ]]; then
            command_plan 'atomically publish recorded content (sync then rename)' "${TARGETS[i]}"
        else command_plan install -D -m "${MODES[i]}" '(rendered content above)' "${TARGETS[i]}"; fi
    done
}
grub_preflight() {
    # Evaluate trusted defaults in the exact order grub-mkconfig uses, replacing
    # our path with the candidate (empty means remove it for rollback).
    bash -e -s -- "$1" "${2-}" <<'GRUB'
source /etc/default/grub
while IFS= read -r file; do
    if [[ $file == /etc/default/grub.d/zzz-vade-kho.cfg ]]; then eval "$1"
    elif [[ -f $file ]]; then source "$file"; fi
done < <(printf '%s\n' /etc/default/grub.d/*.cfg /etc/default/grub.d/zzz-vade-kho.cfg | LC_ALL=C sort -u)
if [[ $2 == require-kho ]]; then
    found=0
    for arg in ${GRUB_CMDLINE_LINUX:-} ${GRUB_CMDLINE_LINUX_DEFAULT:-}; do
        case "$arg" in kho=off) found=1 ;; kho=*) exit 1 ;; esac
    done
    [[ $found == 1 ]]
fi
GRUB
}
kernel_sync_receipt() {
    # No live kernel file changes until its recovery receipt is durable.
    local path=$TX
    # GNU sync -f flushes the containing filesystem, including file content and
    # directory metadata. Include the ancestors in case /var is a separate mount.
    while :; do
        sync -f "$path" || return 1
        [[ $path != / ]] || break
        path=${path%/*}
        [[ -n $path ]] || path=/
    done
}
publication_temporary() { local destination=$1; printf '%s/.%s.vade-%s.tmp\n' "${destination%/*}" "${destination##*/}" "${TX##*/}"; }
publication_cleanup() {
    local temporary=$1 child metadata
    local children=()
    [[ ! -L $temporary ]] || return 1
    [[ -e $temporary ]] || return 0
    # An earlier release used a single regular temporary. Keep its exact-name
    # recovery, while new publications use an exclusively created directory.
    metadata=$(stat -c '%u:%a' "$temporary") || return 1
    if [[ -f $temporary && $metadata == 0:* ]]; then check_path "$temporary"; rm -- "$temporary"; return; fi
    [[ -d $temporary && $metadata == 0:700 ]] || return 1
    check_path "$temporary/content"
    [[ -r $temporary && -x $temporary ]] && ls -A -- "$temporary" >/dev/null || return 1
    children=("$temporary"/* "$temporary"/.[!.]* "$temporary"/..?*)
    for child in "${children[@]}"; do
        [[ -e $child || -L $child ]] || continue
        [[ $child == "$temporary/content" && -f $child && ! -L $child ]] || return 1
        metadata=$(stat -c %u "$child") || return 1
        [[ $metadata == 0 ]] || return 1
    done
    [[ ! -e $temporary/content ]] || rm -- "$temporary/content" || return 1
    rmdir -- "$temporary"
}
kernel_publish() {
    # The adjacent temporary name belongs only to this fresh transaction.
    # Exclusive mkdir avoids following a dangling symlink during file creation.
    local source=$1 destination=$2 mode=${3-} metadata=${4:-$1} temporary
    temporary=$(publication_temporary "$destination") || return 1
    check_path "$temporary/content"
    mkdir -m 700 -- "$temporary" || return 1
    cp -a -- "$source" "$temporary/content" || return 1
    cmp -s "$source" "$temporary/content" || return 1
    chown --reference="$metadata" -- "$temporary/content" || return 1
    if [[ -n $mode ]]; then chmod "$mode" "$temporary/content" || return 1
    else chmod --reference="$metadata" -- "$temporary/content" || return 1; fi
    touch -r "$metadata" -- "$temporary/content" || return 1
    sync -f "$temporary/content" || return 1
    mv -T -- "$temporary/content" "$destination" || return 1
    sync -f "${destination%/*}" || return 1
    rmdir -- "$temporary" || return 1
}
kernel_restore_files() {
    local i dst prior temporary failed=0
    while IFS=$'\t' read -r i dst prior; do
        temporary=$(publication_temporary "$dst")
        # A kill during an earlier publication may leave its adjacent temporary.
        publication_cleanup "$temporary" || { failed=1; continue; }
        if [[ $prior == present ]]; then
            kernel_publish "$TX/prior-$i" "$dst" || failed=1
        else
            rm -f -- "$dst" || failed=1
            [[ ! -d ${dst%/*} ]] || sync -f "${dst%/*}" || failed=1
        fi
    done < "$TX/files.tsv"
    return "$failed"
}
kernel_stage() {
    local i
    kernel_sync_receipt
    for i in "${!TARGETS[@]}"; do
        [[ -d $(dirname "${TARGETS[i]}") ]] || install -d -m 755 "$(dirname "${TARGETS[i]}")"
        kernel_publish "$TX/desired-$i" "${TARGETS[i]}" "${MODES[i]}"
    done
    i=${#TARGETS[@]}
    # grub-mkconfig may fail or be killed; it never writes the live grub.cfg.
    grub-mkconfig -o "$TX/candidate-$i"
    grub-script-check "$TX/candidate-$i"
    mv -- "$TX/candidate-$i" "$TX/desired-$i"
    kernel_sync_receipt
    kernel_publish "$TX/desired-$i" /boot/grub/grub.cfg '' "$TX/prior-$i"
}
rdma_refresh() {
    udevadm control --reload-rules &&
    udevadm trigger --subsystem-match=infiniband_verbs --action=change &&
    udevadm trigger --sysname-match=rdma_cm --action=change &&
    udevadm settle --timeout=30 &&
    systemctl daemon-reload
}
rdma_plan() {
    command_plan udevadm control --reload-rules
    command_plan udevadm trigger --subsystem-match=infiniband_verbs --action=change
    command_plan udevadm trigger --sysname-match=rdma_cm --action=change
    command_plan udevadm settle --timeout=30
    command_plan systemctl daemon-reload
}
network_publish() {
    local destination=${TARGETS[0]} temporary
    temporary=$(publication_temporary "$destination") || return 1
    check_path "$temporary/content"
    [[ -d ${destination%/*} ]] || install -d -m 755 "${destination%/*}" || return 1
    # A different file may have appeared after preflight. Atomic no-overwrite
    # publication protects that file even when another administrator is editing.
    if [[ -e $destination || -L $destination ]]; then
        [[ -f $destination && ! -L $destination ]] && cmp -s "$destination" "$TX/desired-0" || {
            printf 'FAIL netplan destination changed after preflight; preserved\n' >&2
            return 1
        }
    else
        mkdir -m 700 -- "$temporary" || return 1
        install -m 600 -- "$TX/desired-0" "$temporary/content" || return 1
        cmp -s "$TX/desired-0" "$temporary/content" || return 1
        sync -f "$temporary/content" || return 1
        # ln -T never replaces a destination, including a dangling symlink.
        ln -T -- "$temporary/content" "$destination" || return 1
        rm -- "$temporary/content" || return 1
        sync -f "${destination%/*}" || return 1
        rmdir -- "$temporary" || return 1
    fi
}
# Restore only exact files/empty directories this transaction changed. Keep the
# receipt so an operator can retry if a recovery command itself fails.
restore_files() {
    local i dst prior temporary failed=0
    while IFS=$'\t' read -r i dst prior; do
        if [[ $action == network && $dst == /etc/netplan/40-cx7.yaml ]]; then
            temporary=$(publication_temporary "$dst")
            publication_cleanup "$temporary" || { failed=1; continue; }
            # Do not remove a different file that appeared concurrently when
            # exclusive publication refused it. Keep it for administrator review.
            if [[ -e $dst || -L $dst ]]; then
                if [[ -L $dst || ! -f $dst ]] ||
                    { ! cmp -s "$dst" "$TX/desired-$i" && { [[ $prior != present ]] || ! cmp -s "$dst" "$TX/prior-$i"; }; }; then
                    printf 'FAIL netplan destination differs from this transaction; preserved\n' >&2
                    failed=1; continue
                fi
            fi
        fi
        if [[ $prior == present ]]; then cp -a -- "$TX/prior-$i" "$dst" || failed=1
        else rm -f -- "$dst" || failed=1; fi
    done < "$TX/files.tsv"
    return "$failed"
}
restore_rdma() {
    local existed member group_gid node uid gid mode failed=0 entry current_gid memberships accounts status owned=0
    IFS=$'\t' read -r existed member group_gid < "$TX/group.tsv"
    if entry=$(getent group "$RDMA_GROUP"); then
        current_gid=$(cut -d: -f3 <<< "$entry")
        if [[ $existed == yes && $current_gid == "$group_gid" ]]; then owned=1
        elif [[ $existed == no && -s $TX/group-created-gid && $current_gid == "$(cat "$TX/group-created-gid")" ]]; then owned=1
        else failed=1; fi
    else
        status=$?
        [[ $status == 2 && -z $entry && $existed == no ]] || failed=1
    fi
    # A failed groupadd never grants ownership of a preexisting/concurrent group.
    if [[ $owned == 1 && $member == no ]]; then
        if memberships=$(id -nG "$SERVICE_USER"); then
            if tr ' ' '\n' <<< "$memberships" | grep -Fxq "$RDMA_GROUP"; then gpasswd -d "$SERVICE_USER" "$RDMA_GROUP" || failed=1; fi
        else failed=1; fi
    fi
    rdma_refresh || failed=1
    while IFS=$'\t' read -r node uid gid mode; do
        [[ -c $node && ! -L $node ]] || { failed=1; continue; }
        chown "$uid:$gid" "$node" || failed=1
        chmod "$mode" "$node" || failed=1
    done < "$TX/devices.tsv"
    if [[ $existed == no && $owned == 1 ]]; then
        # Do not delete a newly shared group if another administrator used it.
        if entry=$(getent group "$RDMA_GROUP") && [[ -z $(cut -d: -f4 <<< "$entry") ]] &&
            accounts=$(getent passwd) && ! awk -F: -v gid="$current_gid" '$4 == gid { found=1 } END { exit !found }' <<< "$accounts"; then
            groupdel "$RDMA_GROUP" || failed=1
        else failed=1; fi
    fi
    return "$failed"
}
restore_dirs() {
    local dir
    # Parent directories were recorded deepest first. Leave any nonempty one.
    while IFS= read -r dir; do [[ ! -d $dir ]] || rmdir -- "$dir" || return 1; done < "$TX/directories.txt"
}
# A repeated rollback has nothing to activate when every recorded part of the
# prior state is already restored. An uncertain inspection is never a no-op.
prior_identical() {
    local i dst prior dir entry status existed member group_gid memberships node uid gid mode saved_metadata current_metadata temporary
    while IFS=$'\t' read -r i dst prior; do
        if [[ $action == kernel || ( $action == network && $dst == /etc/netplan/40-cx7.yaml ) ]]; then
            temporary="$(dirname "$dst")/.$(basename "$dst").vade-$(basename "$TX").tmp"
            [[ ! -e $temporary ]] || return 1
        fi
        if [[ $prior == present ]]; then
            [[ -f $dst ]] && cmp -s "$TX/prior-$i" "$dst" || return 1
            saved_metadata=$(stat -c '%u:%g:%a' "$TX/prior-$i") || return 1
            current_metadata=$(stat -c '%u:%g:%a' "$dst") || return 1
            [[ $saved_metadata == "$current_metadata" ]] || return 1
        else [[ ! -e $dst ]] || return 1; fi
    done < "$TX/files.tsv"
    while IFS= read -r dir; do [[ ! -e $dir ]] || return 1; done < "$TX/directories.txt"
    if [[ $action == rdma ]]; then
        IFS=$'\t' read -r existed member group_gid < "$TX/group.tsv" || return 1
        if entry=$(getent group "$RDMA_GROUP"); then
            [[ $existed == yes && $(cut -d: -f3 <<< "$entry") == "$group_gid" ]] || return 1
        else status=$?; [[ $status == 2 && -z $entry && $existed == no ]] || return 1; fi
        memberships=$(id -nG "$SERVICE_USER") || return 1
        if tr ' ' '\n' <<< "$memberships" | grep -Fxq "$RDMA_GROUP"; then
            [[ $member == yes ]] || return 1
        else [[ $member == no ]] || return 1; fi
        while IFS=$'\t' read -r node uid gid mode; do
            current_metadata=$(stat -c $'%u\t%g\t%a' "$node") || return 1
            [[ $current_metadata == "$uid"$'\t'"$gid"$'\t'"$mode" ]] || return 1
        done < "$TX/devices.tsv"
    fi
}
recover() {
    local failed=0
    if [[ $action == kernel ]]; then kernel_restore_files || failed=1
    else restore_files || failed=1; fi
    if [[ $action == rdma ]]; then restore_rdma || failed=1; fi
    restore_dirs || failed=1
    return "$failed"
}
on_error() {
    local status=${1:-$?}
    trap - ERR INT TERM
    if [[ -n $TX && -f $TX/ready ]]; then
        printf 'FAIL step interrupted; restoring immediate prior state from %s\n' "$TX" >&2
        if recover; then printf 'PASS immediate prior state restored; receipt retained: %s\n' "$TX" >&2
        else printf 'FAIL recovery incomplete; retry the printed rollback command from the console\n' >&2; fi
        command_plan bash "$SCRIPT_DIR/apply.sh" "$config" rollback "$TX" --apply >&2
    fi
    exit "$status"
}

if [[ $action == rollback ]]; then
    TX=$3
    [[ $TX =~ ^/var/lib/vade-spark-setup/transactions/[A-Za-z0-9._-]+$ && ! -L $TX && -d $TX ]] || die 'use the exact printed transaction directory'
    metadata=$(stat -c '%u:%a' "$TX") || die 'cannot inspect snapshot metadata'
    [[ $metadata == 0:700 ]] || die 'snapshot must be root-owned, mode 700'
    # The transaction's allowlist, group identity and backup files are validated
    # before any restore. Identity and manifest data are never shell code;
    # recorded trusted GRUB defaults use the same evaluation as live defaults.
    bash "$SCRIPT_DIR/transaction-check.sh" "$TX" "$SERVICE_USER" "$RDMA_GROUP"
    read -r action < "$TX/action"
    i=0
    while IFS=$'\t' read -r index dst prior; do
        check_path "$dst"
        [[ $prior != present ]] || file_diff "$dst" "$(cat "$TX/prior-$index")"
        if [[ $prior == absent ]]; then
            [[ ! -f $dst ]] || diff -u --label "$dst (current)" --label "$dst (removed)" "$dst" /dev/null || [[ $? == 1 || ! -e $dst ]]
            command_plan rm -f -- "$dst"
        else command_plan cp -a -- "$TX/prior-$index" "$dst"; fi
        i=$((i+1))
    done < "$TX/files.tsv"
    if [[ $action == network ]]; then
        need ip nmcli netplan ethtool
        bash "$SCRIPT_DIR/network-preflight.sh" "$TX/desired-0" --ssh-only
    elif [[ $action == rdma ]]; then
        need getent id gpasswd groupdel udevadm systemctl chown chmod
        IFS=$'\t' read -r existed member group_gid < "$TX/group.tsv"
        [[ $member == yes ]] || command_plan gpasswd -d "$SERVICE_USER" "$RDMA_GROUP"
        rdma_plan
        while IFS=$'\t' read -r node uid gid mode; do command_plan chown "$uid:$gid" "$node"; command_plan chmod "$mode" "$node"; done < "$TX/devices.tsv"
        [[ $existed == yes ]] || command_plan groupdel "$RDMA_GROUP"
    elif [[ $action == kernel ]]; then
        need bash rm
        restored_kho=
        [[ ! -f $TX/prior-0 ]] || restored_kho=$(cat "$TX/prior-0")
        grub_preflight "$restored_kho" || die 'restored GRUB defaults fail preflight; no files changed'
        command_plan 'atomically restore the recorded prior drop-in and grub.cfg without regeneration'
    fi
    while IFS= read -r dir; do command_plan rmdir -- "$dir"; done < "$TX/directories.txt"
    if prior_identical; then
        result_report unchanged
        echo 'PASS prior state already restored; read-only no-op'
        exit
    fi
    if [[ $apply == 0 ]]; then echo 'PREVIEW recovery only; add --apply to restore'; exit; fi
    exec 9> "$STATE/apply.lock"; flock -n 9 || die 'another setup step is running'
    bash "$SCRIPT_DIR/transaction-check.sh" "$TX" "$SERVICE_USER" "$RDMA_GROUP"
    if [[ $action == network ]]; then bash "$SCRIPT_DIR/network-preflight.sh" "$TX/desired-0" --ssh-only
    elif [[ $action == kernel ]]; then grub_preflight "$restored_kho" || die 'restored GRUB defaults changed; no files restored'; fi
    if prior_identical; then result_report unchanged; echo 'PASS prior state already restored; no-op'; exit; fi
    if ! recover; then die "recovery incomplete; receipt retained: $TX"; fi
    result_report changed
    echo 'PASS prior state restored; restart/reboot only from console when maintenance permits'
    exit
fi

preflight() {
SERVICE_UID=$(id -u "$SERVICE_USER")
SERVICE_GID=$(id -g "$SERVICE_USER")
TARGETS=(); CONTENTS=(); MODES=()
case $action in
packages)
    need apt-get dpkg-query fuser
    # Do not race an existing package transaction; do not stop it.
    for lock in /var/lib/dpkg/lock /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock; do
        if fuser "$lock" >/dev/null 2>&1; then die 'package manager is busy; wait and retry'; fi
    done
    packages=(libibverbs1 ibverbs-providers rdma-core ibverbs-utils perftest iproute2 ethtool openssh-server netplan.io network-manager)
    command_plan apt-get update
    command_plan apt-get install -y "${packages[@]}"
    if [[ $apply == 0 ]]; then echo 'PREVIEW only; add --apply for package installation (OS package transactions are not automatically undone)'; exit; fi
    exec 9> "$STATE/apply.lock" 2>/dev/null || { install -d -m 700 "$STATE"; exec 9> "$STATE/apply.lock"; }
    flock -n 9 || die 'another setup step is running'
    install -d -m 700 "$STATE/transactions"
    TX=$(mktemp -d "$STATE/transactions/packages.XXXXXX")
    dpkg-query -W -f='${Package}\t${Version}\t${db:Status-Status}\n' > "$TX/packages-before.tsv"
    printf 'SNAPSHOT %s (package inventory; use apt/dpkg recovery for package errors)\n' "$TX"
    apt-get update
    apt-get install -y "${packages[@]}"
    result_report changed
    exit ;;
network)
    need ip nmcli netplan ethtool
    add_file /etc/netplan/40-cx7.yaml "$(render_content 40-cx7.yaml)" 600
    # The effective merged netplan view cannot prove ownership of every byte in
    # this destination: another source can hide definitions in this file. Leave
    # all differing existing files intact, including management/other fabrics.
    if [[ -e ${TARGETS[0]} ]] && ! cmp -s "${TARGETS[0]}" <(printf '%s\n' "${CONTENTS[0]}"); then
        die 'existing 40-cx7.yaml differs; preserve the complete file and ask its administrator to review it separately'
    fi
    hosts=$(render_content hosts.fragment)
    merged=$(printf '%s\n' "$hosts" | awk '
        BEGIN { start="# BEGIN VADE SPARK FABRIC"; end="# END VADE SPARK FABRIC" }
        FILENAME=="-" {
            if ($0 !~ /^#/ && NF) aliases[$2]=1
            fragment=fragment (fragments++ ? "\n" : "") $0
            next
        }
        { count=FNR; lines[FNR]=$0; if ($0==start) { starts++; first=FNR }; if ($0==end) { ends++; last=FNR } }
        END {
            if (starts!=ends || starts>1) { print "FAIL malformed managed hosts block" > "/dev/stderr"; exit 1 }
            if (starts && first>=last) { print "FAIL reversed hosts block" > "/dev/stderr"; exit 1 }
            for (i=1; i<=count; i++) if (!starts || i<first || i>last) {
                line=lines[i]; sub(/#.*/, "", line); sub(/^[[:space:]]+/, "", line)
                n=split(line, fields, /[[:space:]]+/)
                for (j=2; j<=n; j++) if (fields[j] in aliases) {
                    print "FAIL fabric alias exists outside managed block" > "/dev/stderr"; exit 1
                }
            }
            for (i=1; i<=count; i++) if (!starts || i<first || i>last) print lines[i]
            print fragment
        }
    ' /etc/hosts -)
    add_file /etc/hosts "$merged" 644
    if files_identical; then result_report unchanged; echo 'PASS identical network and hosts configuration; read-only no-op'; exit; fi
    for i in "${!NETDEVS[@]}"; do
        [[ -d /sys/class/net/${NETDEVS[i]} ]] || die 'configured Ethernet device missing'
        [[ -d /sys/class/infiniband/${RDMA_DEVS[i]}/device/net/${NETDEVS[i]} ]] || die 'configured Ethernet/RDMA mapping does not match'
    done
    printf '%s\n' "${CONTENTS[0]}" | bash "$SCRIPT_DIR/network-preflight.sh" -
    ;;
rdma)
    need getent id groupadd groupdel usermod gpasswd udevadm systemctl chown chmod
    DEVICES=(); DEVICE_STATES=()
    for node in /dev/infiniband/uverbs* /dev/infiniband/rdma_cm; do
        [[ -e $node ]] || continue
        [[ -c $node && ! -L $node ]] || die 'unexpected RDMA device node'
        DEVICES+=("$node")
        device_state=$(stat -c $'%u\t%g\t%a' "$node") || die 'cannot inspect RDMA device metadata'
        DEVICE_STATES+=("$device_state")
    done
    [[ -x /usr/sbin/sshd ]] || die 'install OpenSSH first'
    /usr/sbin/sshd -t
    ssh_settings=$(/usr/sbin/sshd -T) || die 'cannot inspect effective SSH settings'
    ssh_usepam=$(awk '$1=="usepam" {print $2}' <<< "$ssh_settings") || die 'cannot inspect SSH UsePAM'
    [[ $ssh_usepam == yes ]] || die 'SSH UsePAM must be yes'
    grep -Eq '^[[:space:]]*session[[:space:]]+required[[:space:]]+pam_limits\.so' /etc/pam.d/sshd || die 'SSH PAM must include required pam_limits.so'
    unit_state=$(systemctl show ssh.service -p LoadState --value) || die 'cannot inspect ssh.service state'
    [[ $unit_state == loaded ]] || die 'ssh.service is not loaded'
    if [[ -n $SERVICE_UNIT ]]; then
        unit_state=$(systemctl show "$SERVICE_UNIT" -p LoadState --value) || die 'cannot inspect configured service state'
        unit_id=$(systemctl show "$SERVICE_UNIT" -p Id --value) || die 'cannot inspect configured service identity'
        unit_user=$(systemctl show "$SERVICE_UNIT" -p User --value) || die 'cannot inspect configured service user'
        [[ $unit_state == loaded && $unit_id == "$SERVICE_UNIT" ]] || die 'configured Vade service does not exist or is an alias'
        [[ $unit_user == "$SERVICE_USER" ]] || die 'Vade service User must match SERVICE_USER'
        add_file "/etc/systemd/system/$SERVICE_UNIT.d/99-vade-memlock.conf" "$(render_content service-memlock.conf)" 644
    fi
    add_file /etc/security/limits.d/99-vade-memlock.conf "$(render_content 99-vade-memlock.conf)" 644
    add_file /etc/udev/rules.d/99-vade-rdma.rules "$(render_content 99-vade-rdma.rules)" 644
    add_file /etc/systemd/system/ssh.service.d/99-vade-memlock.conf "$(render_content ssh-memlock.conf)" 644
    existed=no; member=no; group_gid=-
    if entry=$(getent group "$RDMA_GROUP"); then
        existed=yes; group_gid=$(cut -d: -f3 <<< "$entry")
        [[ $group_gid =~ ^[0-9]+$ && $(cut -d: -f1 <<< "$entry") == "$RDMA_GROUP" && $entry != *$'\n'* ]] || die 'ambiguous RDMA group identity'
    else
        status=$?
        [[ $status == 2 && -z $entry ]] || die 'cannot inspect RDMA group; no changes are safe'
    fi
    memberships=$(id -nG "$SERVICE_USER") || die 'cannot inspect service account membership'
    if tr ' ' '\n' <<< "$memberships" | grep -Fxq "$RDMA_GROUP"; then member=yes; fi
    if files_identical && [[ $existed == yes && $member == yes ]]; then result_report unchanged; echo 'PASS identical RDMA files and membership; read-only no-op'; exit; fi
    [[ $existed == yes ]] || command_plan groupadd "$RDMA_GROUP"
    [[ $member == yes ]] || command_plan usermod -a -G "$RDMA_GROUP" "$SERVICE_USER"
    rdma_plan ;;
kernel)
    need grub-mkconfig grub-script-check bash mv rm
    [[ $(uname -r) == 7.0.0-1019-nvidia ]] || die 'kernel step is only for known affected 7.0.0-1019-nvidia'
    add_file /etc/default/grub.d/zzz-vade-kho.cfg "$(render_content zzz-vade-kho.cfg)" 644
    check_path /boot/grub/grub.cfg
    [[ -f /boot/grub/grub.cfg ]] || die 'existing /boot/grub/grub.cfg required for recovery'
    metadata=$(stat -c %u /boot/grub/grub.cfg) || die 'cannot inspect grub.cfg ownership'
    [[ $metadata == 0 ]] || die 'existing /boot/grub/grub.cfg must be root-owned for recovery'
    grub_preflight "${CONTENTS[0]}" require-kho || die 'GRUB defaults fail or contain a conflicting kho option; no files changed'
    if files_identical; then result_report unchanged; echo 'PASS identical kernel drop-in; read-only no-op (verify boot state separately)'; exit; fi
    command_plan grub-mkconfig -o '(private transaction candidate)'
    command_plan grub-script-check '(private transaction candidate)'
    command_plan 'record and sync validated grub.cfg candidate before atomic publication' ;;
esac
}
preflight
plan_files
command_plan 'record immediate prior files/absence, created directories and RDMA membership/device permissions in private transaction'
if [[ $apply == 0 ]]; then echo 'PREVIEW only; add --apply after reviewing commands and diffs'; exit; fi

install -d -m 700 "$STATE" "$STATE/transactions"
exec 9> "$STATE/apply.lock"; flock -n 9 || die 'another setup step is running'
# Recompute contents, group membership and device metadata in this process;
# a separate preview would leave stale pre-lock values in the parent.
preflight
plan_files
TX=$(mktemp -d "$STATE/transactions/$action.XXXXXX")
printf '%s\n' "$action" > "$TX/action"
printf '%s\t%s\t%s\t%s\n' "$SERVICE_USER" "$RDMA_GROUP" "$SERVICE_UID" "$SERVICE_GID" > "$TX/identity.tsv"
: > "$TX/files.tsv"; : > "$TX/directories.txt"
for i in "${!TARGETS[@]}"; do
    dst=${TARGETS[i]}; prior=absent
    if [[ -f $dst ]]; then cp -a -- "$dst" "$TX/prior-$i"; prior=present; fi
    printf '%s\n' "${CONTENTS[i]}" > "$TX/desired-$i"
    printf '%s\t%s\t%s\n' "$i" "$dst" "$prior" >> "$TX/files.tsv"
    parent=$(dirname "$dst")
    while [[ ! -e $parent ]]; do
        if ! grep -Fxq "$parent" "$TX/directories.txt"; then printf '%s\n' "$parent" >> "$TX/directories.txt"; fi
        parent=$(dirname "$parent")
    done
done
# Reverse lexical order puts each child before its own parent; destinations may
# share ancestry. No recovery can remove a nonempty directory.
LC_ALL=C sort -ru "$TX/directories.txt" > "$TX/directories.sorted"
mv -- "$TX/directories.sorted" "$TX/directories.txt"
if [[ $action == kernel ]]; then
    i=${#TARGETS[@]}
    cp -a -- /boot/grub/grub.cfg "$TX/prior-$i"
    printf '%s\t%s\tpresent\n' "$i" /boot/grub/grub.cfg >> "$TX/files.tsv"
fi
if [[ $action == rdma ]]; then
    printf '%s\t%s\t%s\n' "$existed" "$member" "$group_gid" > "$TX/group.tsv"
    : > "$TX/devices.tsv"
    for i in "${!DEVICES[@]}"; do
        printf '%s\t%s\n' "${DEVICES[i]}" "${DEVICE_STATES[i]}" >> "$TX/devices.tsv"
    done
fi
printf 'SNAPSHOT %s\n' "$TX"
command_plan bash "$SCRIPT_DIR/apply.sh" "$config" rollback "$TX"
command_plan bash "$SCRIPT_DIR/apply.sh" "$config" rollback "$TX" --apply
# Ready is created only after the entire prior state was saved.
touch "$TX/ready"
trap on_error ERR
trap 'on_error 130' INT
trap 'on_error 143' TERM
if [[ $action == rdma ]]; then
    if [[ $existed == no ]]; then
        groupadd "$RDMA_GROUP"
        entry=$(getent group "$RDMA_GROUP")
        printf '%s\n' "$(cut -d: -f3 <<< "$entry")" > "$TX/group-created-gid"
    fi
    [[ $member == yes ]] || usermod -a -G "$RDMA_GROUP" "$SERVICE_USER"
fi
if [[ $action == kernel ]]; then kernel_stage
else
    if [[ $action == network ]]; then network_publish; fi
    for i in "${!TARGETS[@]}"; do
        [[ $action != network || $i != 0 ]] || continue
        install -D -m "${MODES[i]}" "$TX/desired-$i" "${TARGETS[i]}"
    done
    if [[ $action == rdma ]]; then rdma_refresh; fi
fi
touch "$TX/complete"
result_report changed
trap - ERR INT TERM
printf 'PASS %s staged; no network activation, service restart or reboot performed. Snapshot: %s\n' "$action" "$TX"
