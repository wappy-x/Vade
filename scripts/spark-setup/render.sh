#!/usr/bin/env bash
# Print files and commands by default; local writes require --apply.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
apply=0
if [[ ${!#} == --apply ]]; then apply=1; set -- "${@:1:$#-1}"; fi
[[ $# == 2 ]] || die 'usage: render.sh CONFIG OUTPUT_DIRECTORY [--apply]'
load_config "$1"
output=$2
files=(40-cx7.yaml hosts.fragment 99-vade-memlock.conf ssh-memlock.conf service-memlock.conf 99-vade-rdma.rules zzz-vade-kho.cfg)
contents=()
for file in "${files[@]}"; do
    contents+=("$(render_content "$file")")
    check_path "$output/$file"
done
command_plan mkdir -p -- "$output"
changed=0
for i in "${!files[@]}"; do
    target=$output/${files[i]}
    file_diff "$target" "${contents[i]}"
    if [[ ! -f $target ]] || ! cmp -s "$target" <(printf '%s\n' "${contents[i]}"); then
        changed=1
        file_write_plan "$target" "${contents[i]}"
    fi
done
if [[ $changed == 0 ]]; then echo 'PASS identical rendered files; read-only no-op'; exit; fi
if [[ $apply == 0 ]]; then echo 'PREVIEW only; add --apply to write these local files'; exit; fi
umask 077
mkdir -p -- "$output"
for i in "${!files[@]}"; do
    target=$output/${files[i]}
    [[ ! -f $target ]] || ! cmp -s "$target" <(printf '%s\n' "${contents[i]}") || continue
    printf '%s\n' "${contents[i]}" > "$target"
    chmod 600 "$target"
done
echo 'PASS local files rendered; review before system installation'
