#!/usr/bin/env bash
# Preview by default. Only --apply changes units or saves recovery state.
set -euo pipefail
umask 077
export LC_ALL=C

die() { printf 'FAIL updates: %s\n' "$*" >&2; exit 1; }
[[ $# -ge 1 && $# -le 2 && $1 =~ ^(pause|restore|status)$ ]] || die 'usage: updates.sh pause|restore|status [--apply]'
action=$1
apply=0
if [[ $# == 2 ]]; then [[ $2 == --apply ]] || die 'only --apply is supported'; apply=1; fi
system=$(uname -s) || die 'cannot inspect operating system'
[[ $system == Linux ]] || die 'Linux with systemd is required'
for command in systemctl stat diff; do command -v "$command" >/dev/null || die "$command is required"; done
units=(apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service apt-daily.service apt-daily-upgrade.service)
state_dir=/var/lib/vade-spark-setup
snapshot=$state_dir/update-state.tsv
marker=$state_dir/update-phase

# Read every target before deciding what to change. A package worker is never
# stopped, including one started by a timer during the pause operation.
read_state() {
    local unit=$1 load identity
    load=$(systemctl show --property=LoadState --value "$unit") || die "cannot inspect $unit"
    if [[ $load == not-found ]]; then
        unit_file=absent
        unit_active=$(systemctl show --property=ActiveState --value "$unit") || die "cannot inspect $unit"
        [[ $unit_active == inactive ]] || die "$unit is absent but $unit_active; review manually"
        return
    fi
    [[ $load == loaded || $load == masked ]] || die "$unit has LoadState=$load"
    identity=$(systemctl show --property=Id --value "$unit") || die "cannot inspect $unit identity"
    [[ $identity == "$unit" ]] || die "$unit resolves to another unit"
    unit_file=$(systemctl show --property=UnitFileState --value "$unit") || die "cannot inspect $unit enablement"
    unit_active=$(systemctl show --property=ActiveState --value "$unit") || die "cannot inspect $unit activity"
    [[ $unit_file =~ ^(enabled|enabled-runtime|disabled|static|indirect|masked|masked-runtime)$ ]] || die "$unit has unsupported UnitFileState=$unit_file"
    if [[ $unit_active != active && $unit_active != inactive ]]; then
        [[ ${2-} == allow_updates && $unit =~ ^apt-daily(-upgrade)?\.service$ && $unit_active =~ ^(activating|deactivating)$ ]] || die "$unit is $unit_active; wait for a stable state"
    fi
}
paused_status() {
    local unit result=0
    for unit in "${units[@]}"; do
        read_state "$unit"
        if [[ $unit_file == absent ]]; then
            printf 'PASS updates %s: absent (no unit to pause)\n' "$unit"
        elif [[ $unit_file == masked || $unit_file == masked-runtime ]] && [[ $unit_active == inactive ]]; then
            printf 'PASS updates %s: %s, inactive\n' "$unit" "$unit_file"
        else
            printf 'FAIL updates %s: %s, %s (expected masked and inactive)\n' "$unit" "$unit_file" "$unit_active"
            result=1
        fi
    done
    return "$result"
}
if [[ $action == status ]]; then
    for unit in "${units[@]}"; do
        printf 'COMMAND systemctl show --property=LoadState,Id,UnitFileState,ActiveState %q\n' "$unit"
    done
    echo 'INFO status is read-only; no file differences or proposed system changes'
    paused_status
    exit
fi
if (( apply )); then
    caller_uid=$(id -u) || die 'cannot inspect caller identity'
    [[ $caller_uid == 0 ]] || die '--apply requires root'
    for command in flock install mktemp mv rm; do command -v "$command" >/dev/null || die "$command is required"; done
fi

metadata_value() {
    local value
    value=$(stat -c "$1" -- "$2") || die 'cannot inspect update-state ownership or permissions'
    printf '%s' "$value"
}
link_value() {
    local value
    value=$(readlink -- "$1") || die 'cannot inspect update-unit mask'
    printf '%s' "$value"
}
validate_state_files() {
    local path
    [[ ! -L $state_dir ]] || die 'state directory must not be a symlink'
    if [[ -e $state_dir ]]; then
        [[ -d $state_dir && $(metadata_value %u "$state_dir") == 0 && $(metadata_value %a "$state_dir") == 700 ]] || die 'state directory must be root-owned with mode 700'
        [[ -r $state_dir && -x $state_dir ]] || die 'saved state is not readable; preview as root'
    fi
    for path in "$snapshot" "$marker" "$state_dir/update.lock"; do
        [[ ! -L $path ]] || die 'state files must not be symlinks'
        [[ ! -e $path || ( -f $path && $(metadata_value %u "$path") == 0 && $(metadata_value %a "$path") == 600 ) ]] || die 'state files must be regular, root-owned and mode 600'
        [[ ! -e $path || -r $path ]] || die 'saved state is not readable; preview as root'
    done
}
validate_state_files

# Preserve the unmasked unit files, drop-ins and enablement links by hash. A
# changed package or administrator setting must be detected before unmasking,
# when systemctl's UnitFileState still hides the underlying enablement.
for command in systemd-analyze readlink sha256sum sort ls; do command -v "$command" >/dev/null || die "$command is required"; done
unit_paths=$(systemd-analyze unit-paths) || die 'cannot discover system unit search paths'
[[ -n $unit_paths ]] || die 'system unit search paths are unavailable'
fingerprint_search_paths() {
    local raw root parent existing duplicate
    unit_roots=()
    while IFS= read -r raw; do
        [[ $raw =~ ^/[A-Za-z0-9_./@:+-]+$ && $raw != / ]] || die 'unsupported systemd search path; review manually'
        if [[ ! -e $raw && ! -L $raw ]]; then
            # A failed existence test can also mean an inaccessible ancestor.
            # Prove absence from an inspectable parent before recording it.
            parent=${raw%/*}; parent=${parent:-/}
            while [[ ! -e $parent && ! -L $parent ]]; do
                [[ $parent != / ]] || die 'cannot inspect systemd search path ancestors'
                parent=${parent%/*}; parent=${parent:-/}
            done
            [[ -d $parent && -r $parent && -x $parent ]] || die 'systemd search path ancestor is not inspectable'
            ls -A -- "$parent" >/dev/null || die 'cannot enumerate systemd search path ancestor'
            root=$(readlink -m -- "$raw") || die 'cannot resolve absent systemd search path'
            [[ $root =~ ^/[A-Za-z0-9_./@:+-]+$ && $root != / ]] || die 'unsupported resolved systemd search path'
            [[ ! -e $raw && ! -L $raw ]] || die 'systemd search path appeared during inspection'
            printf '%s\tsearch-path\tabsent\t%s\n' "$raw" "$root"
            continue
        fi
        [[ -d $raw ]] || die 'systemd search path is not a directory'
        root=$(readlink -f -- "$raw") || die 'cannot resolve systemd search path'
        [[ $root =~ ^/[A-Za-z0-9_./@:+-]+$ && $root != / ]] || die 'unsupported resolved systemd search path'
        [[ -d $root && -r $raw && -x $raw && -r $root && -x $root ]] || die 'systemd search path is not inspectable'
        ls -A -- "$raw" >/dev/null || die 'cannot enumerate systemd search path'
        # Include every raw path, even aliases and empty or missing roots, so
        # its appearance or canonical retargeting changes the fingerprint.
        printf '%s\tsearch-path\t%s\n' "$raw" "$root"
        duplicate=0
        for existing in "${unit_roots[@]-}"; do [[ $existing != "$root" ]] || duplicate=1; done
        (( duplicate )) || unit_roots+=("$root")
    done <<<"$unit_paths"
    [[ ${#unit_roots[@]} != 0 ]] || die 'no inspectable system unit search paths'
}

file_digest() {
    local output digest
    output=$(sha256sum -- "$1") || die 'cannot hash underlying unit file'
    digest=${output%% *}
    [[ $digest =~ ^[a-f0-9]{64}$ && $output == "$digest  $1" ]] || die 'ambiguous unit-file hash output'
    printf '%s' "$digest"
}
# Shell globbing and line records are safe only for this deliberately narrow
# filename alphabet. Refuse unsupported names or directory links rather than
# overlook an alias, inherited drop-in or underlying enablement link.
fingerprint_tree() {
    local base=$1 relative=$2 path name child relevant drop target resolved allowed meta digest root
    [[ -r $base && -x $base ]] || die 'systemd directory is not inspectable'
    ls -A -- "$base" >/dev/null || die 'cannot enumerate systemd directory; review manually'
    for path in "$base"/*; do
        name=${path##*/}
        [[ $name =~ ^[A-Za-z0-9_.@:+-]+$ ]] || die 'unsupported systemd object name; review manually'
        child=$relative/$name
        relevant=0
        [[ $name != "$fingerprint_unit" ]] || relevant=1
        for drop in "${fingerprint_dropins[@]}"; do
            case "$child/" in */"$drop"/*) relevant=1;; esac
        done
        target= resolved=
        if [[ -L $path ]]; then
            [[ ! -d $path ]] || die 'symlinked systemd directory needs manual review'
            target=$(readlink -- "$path") || die 'cannot inspect systemd link'
            [[ $target =~ ^[A-Za-z0-9_./@:+-]+$ ]] || die 'unsupported systemd link target; review manually'
            resolved=$(readlink -m -- "$path") || die 'cannot resolve systemd link'
            [[ $resolved =~ ^/[A-Za-z0-9_./@:+-]+$ ]] || die 'unsupported resolved systemd link'
            [[ ${target##*/} != "$fingerprint_unit" && ${resolved##*/} != "$fingerprint_unit" ]] || relevant=1
        fi
        if (( relevant )); then
            # Ignore only the persistent mask created by this helper. Its
            # exact identity and ownership are checked before restoration.
            if [[ $path == /etc/systemd/system/"$fingerprint_unit" && -L $path && $target == /dev/null ]]; then continue; fi
            meta=$(stat -c '%u:%a' -- "$path") || die 'cannot inspect systemd object metadata'
            [[ $meta =~ ^[0-9]+:[0-7]{1,4}$ ]] || die 'unsupported systemd object metadata'
            if [[ -L $path ]]; then
                allowed=0
                [[ $resolved != /dev/null ]] || allowed=1
                for root in "${unit_roots[@]}"; do
                    [[ $resolved != "$root" && $resolved != "$root/"* ]] || allowed=1
                done
                (( allowed )) || die 'unit link target outside systemd paths needs manual review'
                digest=
                if [[ $resolved != /dev/null && -e $path ]]; then
                    [[ -f $path ]] || die 'unsupported linked systemd object'
                    digest=$(file_digest "$path") || return 1
                fi
                printf '%s\t%s\tlink\t%s\t%s\n' "$path" "$meta" "$target" "$digest"
            elif [[ -f $path ]]; then
                digest=$(file_digest "$path") || return 1
                printf '%s\t%s\tfile\t%s\n' "$path" "$meta" "$digest"
            elif [[ -d $path ]]; then
                printf '%s\t%s\tdirectory\n' "$path" "$meta"
            else die 'unsupported systemd object; review manually'; fi
        fi
        if [[ ! -L $path && -d $path ]]; then fingerprint_tree "$path" "$child" || return 1; fi
    done
}
fingerprint() {
    local fingerprint_unit=$1 kind stem remaining root output digest
    local unit_roots=()
    local fingerprint_dropins=("$1.d" "${1##*.}.d")
    kind=${1##*.}; stem=${1%.*}; remaining=$stem
    while [[ $remaining == *-* ]]; do
        remaining=${remaining#*-}
        fingerprint_dropins+=("${stem%"$remaining"}.$kind.d")
    done
    output=$(
        shopt -s nullglob dotglob
        {
            # Resolve and validate the raw paths on every call, including the
            # preflight under the lock; never reuse earlier canonical roots.
            fingerprint_search_paths || exit 1
            for root in "${unit_roots[@]}"; do fingerprint_tree "$root" '' || exit 1; done
        } | sort | sha256sum
    ) || die 'cannot inspect underlying unit files'
    digest=${output%% *}
    [[ $digest =~ ^[a-f0-9]{64}$ && $output == "$digest  -" ]] || die 'ambiguous unit fingerprint'
    printf 'shell-v1:%s\n' "$digest"
}

saved_file=() saved_active=() saved_hash=()
phase=pausing
new_snapshot=0
snapshot_text=
load_saved() {
    local unit saved_unit file active digest extra row row_mode=
    saved_file=() saved_active=() saved_hash=()
    if [[ ! -e $snapshot ]]; then
        [[ $action == pause && ! -e $marker ]] || die 'no complete saved pause state'
        new_snapshot=1
        return
    fi
    exec 8<"$snapshot"
    for unit in "${units[@]}"; do
        IFS= read -r row <&8 || die 'incomplete snapshot'
        IFS=$'\t' read -r saved_unit file active digest extra <<<"$row"
        if [[ -n $digest ]]; then
            [[ $row == "$saved_unit"$'\t'"$file"$'\t'"$active"$'\t'"$digest" && $row_mode != legacy ]] || die 'ambiguous snapshot fields'
            row_mode=hashed
        else
            [[ $row == "$saved_unit"$'\t'"$file"$'\t'"$active" && $row_mode != hashed ]] || die 'ambiguous snapshot fields'
            row_mode=legacy
        fi
        [[ $saved_unit == "$unit" && -z $extra ]] || die 'ambiguous snapshot row'
        [[ $file =~ ^(absent|enabled|enabled-runtime|disabled|static|indirect|masked|masked-runtime)$ ]] || die 'unsupported saved unit state'
        [[ $active == active || $active == inactive ]] || die 'unsupported saved active state'
        [[ $file != absent || $active == inactive ]] || die 'invalid absent unit snapshot'
        [[ $file != masked && $file != masked-runtime || $active == inactive ]] || die 'invalid active masked unit snapshot'
        [[ $digest =~ ^shell-v1:[a-f0-9]{64}$ ]] || die 'legacy or unsupported fingerprint format; preserve the snapshot and ask an administrator to restore manually'
        saved_file+=("$file") saved_active+=("$active") saved_hash+=("$digest")
    done
    if IFS= read -r extra <&8; then die 'unexpected extra snapshot row'; fi
    exec 8<&-
    if [[ -e $marker ]]; then phase=$(cat "$marker"); fi
    [[ $phase =~ ^(pausing|paused|restoring|restored)$ ]] || die 'unknown saved phase'
}
load_saved
current_file=() current_active=() current_hash=()
preflight() {
    local i unit exec_start exec_pattern path digest
    current_file=() current_active=() current_hash=()
    for i in "${!units[@]}"; do
        unit=${units[i]}
        read_state "$unit" "${1-exact}"
        current_file+=("$unit_file") current_active+=("$unit_active")
        digest=$(fingerprint "$unit") || die "cannot fingerprint $unit"
        [[ $digest =~ ^shell-v1:[a-f0-9]{64}$ ]] || die "ambiguous fingerprint for $unit"
        current_hash+=("$digest")
        if [[ $unit =~ ^apt-daily(-upgrade)?\.service$ && $unit_active != inactive ]]; then
            [[ $action == restore && ( $phase == restored || ${1-} == allow_updates ) ]] || die "$unit is running; let the package transaction finish, then retry (nothing is killed)"
        fi
        if [[ $unit == unattended-upgrades.service && ( $unit_active == active || ( $action == restore && ${saved_active[i]-} == active ) ) ]]; then
            exec_start=$(systemctl show --property=ExecStart --value "$unit") || die 'cannot inspect unattended-upgrades helper'
            # A masked unit has no ExecStart; its recorded file hash proves the
            # validated shutdown helper has not changed since pause.
            if [[ $unit_file != masked && $unit_file != masked-runtime ]]; then
                # Recognize one exact command/argv and a narrow, single-entry
                # systemctl format. Extra entries, arguments, fields or opaque
                # encodings cannot establish identity safely in shell.
                exec_pattern='^\{ path=/usr/share/unattended-upgrades/unattended-upgrade-shutdown ; argv\[\]=/usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal ; ignore_errors=no ; start_time=\[[A-Za-z0-9 :+./-]+\] ; stop_time=\[[A-Za-z0-9 :+./-]+\] ; pid=[0-9]+ ; code=(\(null\)|exited|killed|dumped) ; status=[0-9]+/[A-Za-z0-9]+ \}$'
                [[ $exec_start =~ $exec_pattern ]] || die 'cannot safely identify the single Ubuntu shutdown helper from ExecStart; preserve state and review manually'
            fi
        fi
        if (( new_snapshot )); then
            [[ $unit_file != masked && $unit_file != masked-runtime || $unit_active == inactive ]] || die "$unit is masked but active"
            saved_file+=("$unit_file") saved_active+=("$unit_active") saved_hash+=("${current_hash[i]}")
        else
            [[ $unit_file == "${saved_file[i]}" || ( ${saved_file[i]} != absent && ${saved_file[i]} != masked && ${saved_file[i]} != masked-runtime && $unit_file == masked ) ]] || die "$unit state changed outside this script"
            [[ $unit_active == inactive || ( ${saved_active[i]} == active && $unit_active == active ) || ( $action == restore && ( $phase == restored || ${1-} == allow_updates ) && $unit =~ ^apt-daily(-upgrade)?\.service$ ) ]] || die "$unit became active outside this script"
            if [[ -n ${saved_hash[i]-} ]]; then
                [[ ${current_hash[i]} == "${saved_hash[i]}" ]] || die "$unit underlying files or enablement changed; preserve the snapshot and review manually"
            elif [[ $unit_file != "${saved_file[i]}" ]]; then
                die 'legacy snapshot lacks underlying unit hashes; preserve it and ask an administrator to restore manually'
            fi
        fi
        if [[ $action == pause && $unit_file != absent && $unit_file != masked && $unit_file != masked-runtime ]]; then
            path=/etc/systemd/system/$unit
            [[ ! -e $path && ! -L $path ]] || die "$unit has a local unit file that systemctl mask would replace; review manually"
        fi
        if [[ $action == restore && $unit_file == masked && ${saved_file[i]} != masked ]]; then
            path=/etc/systemd/system/$unit
            [[ -L $path && $(link_value "$path") == /dev/null && $(metadata_value %u "$path") == 0 ]] || die "$unit persistent mask is not the expected root-owned symlink"
        fi
    done
    if [[ $action == pause ]]; then
        [[ $phase != restoring ]] || die 'complete restore before pausing again'
        if [[ $phase == restored ]]; then
            for i in "${!units[@]}"; do
                [[ ${current_file[i]} == "${saved_file[i]}" && ${current_active[i]} == "${saved_active[i]}" ]] || die 'state changed since restore; preserve the snapshot and review manually'
            done
        fi
    fi
}
if [[ $action == restore && $phase == restored ]]; then preflight allow_updates; else preflight; fi
if (( new_snapshot )); then
    for i in "${!units[@]}"; do
        snapshot_text+=$(printf '%s\t%s\t%s\t%s' "${units[i]}" "${saved_file[i]}" "${saved_active[i]}" "${saved_hash[i]}")$'\n'
    done
else
    snapshot_text=$(cat "$snapshot")$'\n'
fi

print_command() { printf '  '; printf '%q ' "$@"; printf '\n'; }
print_diff() {
    local path=$1 contents=$2 prior=/dev/null result=0
    [[ ! -e $path ]] || prior=$path
    diff -u --label "$path (prior)" --label "$path (proposed)" "$prior" <(printf '%s' "$contents") || result=$?
    [[ $result -le 1 ]] || die "cannot compare $path"
}
changed=0
if [[ $action == pause ]]; then
    [[ $phase == paused ]] || changed=1
    for i in "${!units[@]}"; do
        [[ ${current_file[i]} == absent || ( ${current_file[i]} =~ ^masked(-runtime)?$ && ${current_active[i]} == inactive ) ]] || changed=1
    done
else
    [[ $phase == restored ]] || changed=1
fi
if (( ! changed )); then
    printf 'PASS updates: %s already complete; read-only no-op\n' "$action"
    exit
fi
printf 'Proposed %s; all targets and saved state passed preflight.\n' "$action"
[[ -e $state_dir ]] || print_command install -d -m 700 "$state_dir"
printf '  Acquire exclusive lock: %s/update.lock\n' "$state_dir"
if (( new_snapshot )); then
    print_diff "$snapshot" "$snapshot_text"
fi
if [[ $action == pause ]]; then start_phase=pausing; target_phase=paused; else start_phase=restoring; target_phase=restored; fi
print_diff "$marker" "$start_phase"$'\n'
print_diff "$marker" "$target_phase"$'\n'
for i in "${!units[@]}"; do
    if [[ $action == pause && ${current_file[i]} != absent && ${current_file[i]} != masked && ${current_file[i]} != masked-runtime ]]; then print_command systemctl mask -- "${units[i]}"; fi
    if [[ $action == restore && ${current_file[i]} == masked && ${saved_file[i]} != masked ]]; then print_command systemctl unmask -- "${units[i]}"; fi
done
if [[ $action == pause ]]; then
    for unit in apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service; do
        for i in "${!units[@]}"; do [[ ${units[i]} != "$unit" || ${current_file[i]} == absent ]] || print_command systemctl stop -- "$unit"; done
    done
else
    for type in service timer; do
        for i in "${!units[@]}"; do
            [[ ${units[i]} != *.$type || ${saved_active[i]} != active || ${current_active[i]} == active ]] || print_command systemctl start -- "${units[i]}"
        done
    done
fi
if (( ! apply )); then printf 'Preview only; rerun the same command with --apply to make these changes.\n'; exit; fi

# No directory, lock, temporary file or snapshot is created until every target
# has passed preflight. Recheck under the lock before publishing recovery state.
expected_file=("${current_file[@]}") expected_active=("${current_active[@]}") expected_hash=("${current_hash[@]}")
expected_snapshot=$snapshot_text expected_phase=$phase
[[ -e $state_dir ]] || install -d -m 700 "$state_dir"
exec 9>>"$state_dir/update.lock"
flock -n 9 || die 'another updates.sh command is running'
validate_state_files
# Another invocation could have created a snapshot while this one waited.
if (( new_snapshot )); then [[ ! -e $snapshot && ! -e $marker ]] || die 'saved state appeared; rerun to inspect it'; fi
new_snapshot=0
preflight
for i in "${!units[@]}"; do
    [[ ${current_file[i]} == "${expected_file[i]}" && ${current_active[i]} == "${expected_active[i]}" && ${current_hash[i]} == "${expected_hash[i]}" ]] || die 'target state changed after preflight; rerun'
done
if [[ -e $snapshot ]]; then
    [[ $(cat "$snapshot")$'\n' == "$expected_snapshot" ]] || die 'snapshot changed after preflight'
    [[ ! -e $marker || $(cat "$marker") == "$expected_phase" ]] || die 'phase changed after preflight'
fi
pending=
trap '[[ -z $pending ]] || rm -f -- "$pending"' EXIT
write_state() {
    pending=$(mktemp "$state_dir/.update-state.XXXXXX")
    printf '%s' "$2" >"$pending"
    mv -- "$pending" "$1"
    pending=
}
if [[ ! -e $snapshot ]]; then
    # One atomic snapshot contains both original state and its file hashes.
    write_state "$snapshot" "$snapshot_text"
fi
write_state "$marker" "$start_phase"$'\n'
if [[ $action == pause ]]; then
    for i in "${!units[@]}"; do
        [[ ${current_file[i]} == absent || ${current_file[i]} == masked || ${current_file[i]} == masked-runtime ]] || systemctl mask -- "${units[i]}"
    done
    for i in 0 1; do [[ ${current_file[i]} == absent ]] || systemctl stop -- "${units[i]}"; done
    for unit in apt-daily.service apt-daily-upgrade.service; do
        read_state "$unit"
        [[ $unit_active == inactive ]] || die "$unit started during pause; let it finish, then retry pause --apply (nothing is killed)"
    done
    [[ ${current_file[2]} == absent ]] || systemctl stop -- unattended-upgrades.service
    paused_status
else
    for i in "${!units[@]}"; do
        [[ ${current_file[i]} != masked || ${saved_file[i]} == masked ]] || systemctl unmask -- "${units[i]}"
    done
    # Restore targets before starting timers, which may run missed updates.
    for i in "${!units[@]}"; do
        read_state "${units[i]}"
        [[ $unit_file == "${saved_file[i]}" ]] || die 'enablement changed during restore; preserve state and review manually'
    done
    for type in service timer; do
        for i in "${!units[@]}"; do
            [[ ${units[i]} != *.$type || ${saved_active[i]} != active || ${current_active[i]} == active ]] || systemctl start -- "${units[i]}"
        done
    done
    preflight allow_updates
    for i in "${!units[@]}"; do
        [[ ${current_file[i]} == "${saved_file[i]}" ]] || die 'restored enablement differs from the snapshot'
        if [[ ${units[i]} =~ ^apt-daily(-upgrade)?\.service$ && ${saved_active[i]} == inactive && ${current_active[i]} != inactive ]]; then
            printf 'INFO %s: %s after timer restoration; let the package transaction finish\n' "${units[i]}" "${current_active[i]}"
        else
            [[ ${current_active[i]} == "${saved_active[i]}" ]] || die 'restored activity differs from the snapshot'
        fi
    done
fi
write_state "$marker" "$target_phase"$'\n'
printf 'PASS updates: %s complete; original unit state retained in %s\n' "$action" "$snapshot"
