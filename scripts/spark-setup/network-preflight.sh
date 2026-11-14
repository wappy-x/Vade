#!/usr/bin/env bash
# Read-only fabric guard. Accept only understood line-oriented system output.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
export LC_ALL=C
[[ $# == 1 || ( $# == 2 && $2 == --ssh-only ) ]] || die 'usage: network-preflight.sh DESIRED|- [--ssh-only]'
ssh_only=${2-}
for program in timeout ip awk sed nmcli netplan ethtool ls; do
    command -v "$program" >/dev/null || die "missing read-only prerequisite: $program"
done
# Capture privately; never echo network definitions or command diagnostics.
inspect() { timeout 20 "$@" 2>/dev/null || die "cannot inspect $1; no network change is safe"; }
if [[ $1 == - ]]; then desired=$(cat)
else [[ -f $1 && ! -L $1 ]] || die 'desired fabric file must be regular'; desired=$(cat -- "$1"); fi
# This is the renderer's exact subset, not a YAML parser. Extra keys, changed
# indentation, selectors, aliases, tags and alternate encodings are refused.
targets=$(awk '
function bad() { failed=1; exit 1 }
function ipv4(value, p, i) {
    if (split(value,p,".") != 4) return 0
    for(i=1;i<=4;i++) if(p[i] !~ /^(0|[1-9][0-9]*)$/ || length(p[i])>3 || p[i]+0>255) return 0
    return p[4]+0>0 && p[4]+0<255
}
NR==1 { if($0 != "network:") bad(); next }
NR==2 { if($0 != "  version: 2") bad(); next }
NR==3 { if($0 != "  ethernets:") bad(); next }
{
    part=(NR-4)%6
    if(part==0) {
        if($0 !~ /^    [A-Za-z_][A-Za-z0-9_.-]*:$/) bad()
        name=$0; sub(/^    /,"",name); sub(/:$/,"",name)
        if(name ~ /^(null|Null|NULL|true|True|TRUE|false|False|FALSE|yes|Yes|YES|no|No|NO|on|On|ON|off|Off|OFF|y|Y|n|N)$/ || seen[name]++) bad()
        names[++count]=name
    } else if(part==1 && $0 != "      renderer: NetworkManager") bad()
    else if(part==2 && $0 != "      dhcp4: false") bad()
    else if(part==3 && $0 != "      dhcp6: false") bad()
    else if(part==4 && $0 != "      mtu: 9000") bad()
    else if(part==5) {
        if($0 !~ /^      addresses: \[[0-9.]+\/24\]$/) bad()
        address=$0; sub(/^      addresses: \[/,"",address); sub(/\/24\]$/,"",address)
        if(!ipv4(address,p)) bad()
    }
}
END { if(failed || NR<3 || (NR-3)%6 || (count!=2 && count!=4)) exit 1; for(i=1;i<=count;i++) print names[i] }
' <<< "$desired") || die 'desired fabric is outside the exact supported rendered format'
target_words=$(tr '\n' ' ' <<< "$targets")

raw_links=$(inspect ip -o link show)
# Resolve both name@parent and master relationships before testing SSH paths.
# Unknown identities/relationships refuse the whole operation.
links=$(awk -v targets="$target_words" '
function bad() { failed=1; exit 1 }
function mac(value, p, i) {
    if(split(value,p,":")!=6) return 0
    for(i=1;i<=6;i++) if(p[i] !~ /^[0-9a-fA-F][0-9a-fA-F]$/) return 0
    return 1
}
BEGIN { n=split(targets,t," "); for(i=1;i<=n;i++) wanted[t[i]]=1 }
{
    if($1 !~ /^[1-9][0-9]*:$/ || $2 !~ /^[A-Za-z0-9_.@-]+:$/ || $3 !~ /^<[^>]*>$/ || $4!="mtu" || $5 !~ /^[1-9][0-9]*$/) bad()
    if_index=$1; sub(/:$/,"",if_index); value=$2; sub(/:$/,"",value)
    pair_count=split(value,pair,"@"); name=pair[1]
    if(if_index+0<1 || index_seen[if_index]++ || name_seen[name]++ || pair_count>2 || (pair_count==2 && !pair[2])) bad()
    by_index[if_index]=name; indices[name]=if_index; parent[name]=pair[2]; names[++count]=name
    type[name]=""; address[name]="-"
    for(i=3;i<=NF;i++) {
        if($i=="master") { if(master[name] || i==NF || $(i+1) !~ /^[A-Za-z0-9_.-]+$/) bad(); master[name]=$(i+1) }
        if($i ~ /^link\//) { if(type[name] || i==NF) bad(); type[name]=$i; sub(/^link\//,"",type[name]); address[name]=tolower($(i+1)) }
    }
    if(!type[name]) bad()
    if(wanted[name] && (type[name]!="ether" || !mac(address[name],m))) bad()
}
END {
    if(failed || !count) exit 1
    for(name in wanted) if(wanted[name] && !name_seen[name]) exit 1
    for(i=1;i<=count;i++) {
        name=names[i]
        for(j=1;j<=2;j++) {
            other=(j==1 ? parent[name] : master[name]); if(!other) continue
            if(other ~ /^if[0-9]+$/ && !name_seen[other]) { sub(/^if/,"",other); other=by_index[other] }
            if(!other || !name_seen[other]) exit 1
            if(other!=name) { edge[name SUBSEP other]=1; edge[other SUBSEP name]=1; has_edge[name]=has_edge[other]=1 }
        }
    }
    for(name in wanted) if(wanted[name]) related[name]=1
    do {
        changed=0
        for(edge_key in edge) { split(edge_key,p,SUBSEP); if(related[p[1]] && !related[p[2]]) { related[p[2]]=1; changed=1 } }
    } while(changed)
    for(i=1;i<=count;i++) { name=names[i]; print name, type[name], address[name], (related[name]?"related":"clear"), (has_edge[name]?"connected":"standalone"), indices[name] }
}' <<< "$raw_links") || die 'cannot resolve live Ethernet identities and relationships'
name_exists() { awk -v name="$1" '$1==name { found=1 } END { exit !found }' <<< "$links"; }
related_name() { awk -v name="$1" '$1==name && $4=="related" { found=1 } END { exit !found }' <<< "$links"; }
raw_addresses=$(inspect ip -o address show)
addresses=$(awk '
function bad() { failed=1; exit 1 }
function ipv4(value,p,i) {
    if(split(value,p,".")!=4) return 0
    for(i=1;i<=4;i++) if(p[i] !~ /^(0|[1-9][0-9]*)$/ || length(p[i])>3 || p[i]+0>255) return 0
    return 1
}
function ipv6(value,p,n,i,compressed,groups,copy) {
    if(value !~ /^[0-9a-fA-F:]+$/ || value ~ /:::/ || value ~ /^:[^:]/ || value ~ /[^:]:$/) return 0
    copy=value; compressed=gsub(/::/,"X",copy)
    if(compressed>1) return 0
    n=split(value,p,":"); groups=0
    for(i=1;i<=n;i++) { if(p[i]=="" && compressed) continue; if(p[i] !~ /^[0-9a-fA-F]+$/ || length(p[i])>4) return 0; groups++ }
    return (compressed && groups<8) || (!compressed && groups==8)
}
/^[[:space:]]*$/ { next }
{
    if($1 !~ /^[1-9][0-9]*:$/ || $2 !~ /^[A-Za-z0-9_.@-]+$/ || ($3!="inet" && $3!="inet6")) bad()
    name=$2; sub(/@.*$/,"",name)
    if(split($4,cidr,"/")!=2 || cidr[2] !~ /^(0|[1-9][0-9]*)$/) bad()
    if($3=="inet" && (!ipv4(cidr[1],p) || cidr[2]+0>32)) bad()
    if($3=="inet6" && (!ipv6(cidr[1],p) || cidr[2]+0>128)) bad()
    if_index=$1; sub(/:$/,"",if_index); print name, $3, tolower(cidr[1]), if_index
}
END { if(failed) exit 1 }
' <<< "$raw_addresses") || die 'cannot interpret live interface addresses'
while read -r name family address if_index; do
    [[ -n $name ]] || continue
    awk -v name="$name" -v if_index="$if_index" '$1==name && $6==if_index { found=1 } END { exit !found }' <<< "$links" || die 'address refers to an unknown or inconsistent live interface'
done <<< "$addresses"

protect_session() {
    local remote remote_port local_address local_port extra local_names route route_name pid seen= comm parent tty_name console=0
    if [[ -z ${SSH_CONNECTION:-} ]]; then
        [[ -z ${SSH_CLIENT:-} && -z ${SSH_TTY:-} ]] || die 'SSH provenance is missing; preserve SSH_CONNECTION through sudo'
        for descriptor in 0 1 2; do
            tty_name=$(tty <&"$descriptor" 2>/dev/null) || continue
            [[ $tty_name =~ ^/dev/tty([1-9]|[1-5][0-9]|6[0-3])$ ]] && console=1
        done
        [[ $console == 1 ]] || die 'connection provenance unavailable; use a real Linux console or preserve SSH_CONNECTION'
        pid=$PPID
        while [[ $pid != 0 ]]; do
            [[ $pid =~ ^[0-9]+$ && $seen != *" $pid "* && -r /proc/$pid/comm && -r /proc/$pid/status ]] || die 'cannot establish local console process provenance'
            seen+=" $pid "; comm=$(cat "/proc/$pid/comm" 2>/dev/null) || die 'cannot read local console process provenance'
            [[ $comm != sshd* && $comm != dropbear* ]] || die 'SSH login lost provenance; preserve SSH_CONNECTION through sudo'
            parent=$(awk '$1=="PPid:" && $2 ~ /^[0-9]+$/ && NF==2 { count++; value=$2 } END { if(count!=1) exit 1; print value }' "/proc/$pid/status") || die 'cannot identify local console ancestry'
            pid=$parent
        done
        return
    fi
    read -r remote remote_port local_address local_port extra <<< "$SSH_CONNECTION"
    [[ -n $remote && -n $local_address && -z $extra && $SSH_CONNECTION != *$'\n'* ]] || die 'cannot interpret SSH_CONNECTION'
    [[ $remote != *:* && $local_address != *:* ]] || die 'IPv6 SSH ownership needs a structured address parser; use IPv4 management or a real Linux console'
    awk -v remote="$remote" -v local_ip="$local_address" -v rp="$remote_port" -v lp="$local_port" '
    function ipv4(value,p,i) { if(split(value,p,".")!=4) return 0; for(i=1;i<=4;i++) if(p[i] !~ /^(0|[1-9][0-9]*)$/ || length(p[i])>3 || p[i]+0>255) return 0; return 1 }
    BEGIN { exit !(ipv4(remote,p) && ipv4(local_ip,p) && rp ~ /^[0-9]+$/ && lp ~ /^[0-9]+$/ && length(rp)<=5 && length(lp)<=5 && rp+0>=1 && rp+0<=65535 && lp+0>=1 && lp+0<=65535) }
    ' || die 'cannot interpret SSH connection addresses or ports'
    local_names=$(awk -v address="$local_address" '$2=="inet" && $3==address { print $1 }' <<< "$addresses")
    [[ -n $local_names && $local_names != *$'\n'* ]] || die 'SSH local address does not identify one live interface'
    route=$(inspect ip -o -4 route get "$remote" from "$local_address")
    route_name=$(awk '
    NF { lines++; for(i=1;i<=NF;i++) if($i=="dev") { if(i==NF) exit 1; count++; name=$(i+1) } }
    END { if(lines!=1 || count!=1 || name !~ /^[A-Za-z0-9_.-]+$/) exit 1; print name }
    ' <<< "$route") || die 'cannot identify the route carrying the SSH session'
    name_exists "$route_name" || die 'SSH route refers to an unknown live interface'
    if related_name "$local_names" || related_name "$route_name"; then die 'a fabric interface carries or underlies the current SSH session'; fi
}
protect_session
if [[ $ssh_only == --ssh-only ]]; then
    echo 'PASS rollback interface identity and SSH path are safe'; exit
fi
while IFS= read -r name; do
    awk -v name="$name" '$1==name && $5=="standalone" { found=1 } END { exit !found }' <<< "$links" || die 'a fabric interface already has a bridge, bond, VLAN or parent relationship'
done <<< "$targets"
awk -v targets="$target_words" '
BEGIN { n=split(targets,t," "); for(i=1;i<=n;i++) wanted[t[i]]=1 }
wanted[$1] && ($2!="inet6" || $3 !~ /^fe[89ab][0-9a-f]:/) { exit 1 }
' <<< "$addresses" || die 'a fabric interface already has a configured address'
for family in -4 -6; do
    routes=$(inspect ip -o "$family" route show table all)
    awk -v targets="$target_words" -v family="$family" '
    function ipv4(value,p,i) {
        if(split(value,p,".")!=4) return 0
        for(i=1;i<=4;i++) if(p[i] !~ /^(0|[1-9][0-9]*)$/ || length(p[i])>3 || p[i]+0>255) return 0
        return 1
    }
    function ipv6(value,p,n,i,compressed,groups,copy) {
        if(value !~ /^[0-9a-fA-F:]+$/ || value ~ /:::/ || value ~ /^:[^:]/ || value ~ /[^:]:$/) return 0
        copy=value; compressed=gsub(/::/,"X",copy); if(compressed>1) return 0
        n=split(value,p,":"); groups=0
        for(i=1;i<=n;i++) { if(p[i]=="" && compressed) continue; if(p[i] !~ /^[0-9a-fA-F]+$/ || length(p[i])>4) return 0; groups++ }
        return (compressed && groups<8) || (!compressed && groups==8)
    }
    BEGIN { n=split(targets,t," "); for(i=1;i<=n;i++) wanted[t[i]]=1 }
    NF {
        relevant=0; gateway=0; multipath=0; kernel=0; device_count=0
        destination=$1
        if(destination ~ /^(unicast|local|broadcast|multicast|blackhole|unreachable|prohibit|throw|nat)$/) destination=$2
        fields=split(destination,cidr,"/"); maximum=(family=="-4" ? 32 : 128)
        prefix=(fields==2 ? cidr[2]+0 : maximum)
        if(destination!="default" && (fields>2 || (fields==2 && (cidr[2] !~ /^(0|[1-9][0-9]*)$/ || prefix>maximum)) || (family=="-4" ? !ipv4(cidr[1],address_parts) : !ipv6(cidr[1],address_parts)))) exit 1
        for(i=1;i<=NF;i++) {
            if($i=="dev") { if(i==NF || $(i+1) !~ /^[A-Za-z0-9_.-]+$/) exit 1; device_count++; if(wanted[$(i+1)]) relevant=1 }
            if($i=="via") gateway=1
            if($i=="nexthop") multipath=1
            if($i=="nhid" || $i=="encap") exit 1
            if($i=="proto" && $(i+1)=="kernel") kernel=1
        }
        if(!relevant) next
        harmless=(family=="-6" && kernel && !gateway && !multipath && device_count==1 && ((cidr[1] ~ /^fe[89ab][0-9a-f]:[0-9a-f:]*$/ && prefix>=10 && prefix<=128) || (cidr[1] ~ /^ff[0-9a-f][0-9a-f]:[0-9a-f:]*$/ && prefix>=8 && prefix<=128)))
        if(!harmless) exit 1
    }
    ' <<< "$routes" || die 'a fabric interface already carries a route, or route inspection is ambiguous'
done

# netplan get merges /lib, /etc and /run. Only inspect Ethernet ownership and
# device-section presence; never dump wifi credentials or native keyfile data.
empty_section() { [[ $1 == '' || $1 == null || $1 == '{}' ]]; }
ethernet=$(inspect netplan get network.ethernets)
if ! empty_section "$ethernet"; then
    awk -v targets="$target_words" '
    BEGIN { n=split(targets,t," "); for(i=1;i<=n;i++) wanted[t[i]]=1 }
    /^[A-Za-z_][A-Za-z0-9_.-]*:$/ {
        name=$0; sub(/:$/,"",name); if(wanted[name] || seen[name]++) exit 1
        if(name ~ /^(null|Null|NULL|true|True|TRUE|false|False|FALSE|yes|Yes|YES|no|No|NO|on|On|ON|off|Off|OFF|y|Y|n|N)$/) exit 1
        current=name; subsection=""; next
    }
    {
        if(!current || $0 ~ /[&*!{}\[\]"\047\t]/ || $0 ~ /(^|[[:space:]])(match|set-name|networkmanager|<<):/) exit 1
        if($0 ~ /^  (renderer|dhcp4|dhcp6|mtu|optional|accept-ra|wakeonlan|gateway4|gateway6): [A-Za-z0-9_.:\/+ -]+$/) { subsection=""; next }
        if($0 ~ /^  (addresses|link-local|optional-addresses):$/) { subsection="list"; next }
        if($0 ~ /^  nameservers:$/) { subsection="nameservers"; next }
        if(subsection=="nameservers" && $0 ~ /^    (addresses|search):$/) next
        if((subsection=="list" && $0 ~ /^  (  )?- [A-Za-z0-9_.:\/+ -]+$/) || (subsection=="nameservers" && $0 ~ /^    (  )?- [A-Za-z0-9_.:\/+ -]+$/)) next
        exit 1
    }
    END { if(!current) exit 1 }
    ' <<< "$ethernet" || die 'effective netplan ownership is present or outside the safe plain Ethernet subset; selectors and complex YAML need owner review'
fi
for section in bridges bonds vlans nm-devices tunnels vrfs virtual-ethernets dummy-devices; do
    value=$(inspect netplan get "network.$section")
    empty_section "$value" || die 'effective netplan virtual or opaque device ownership needs structured parsing; leave the fabric unchanged'
done

nm_value() {
    local value
    value=$(inspect nmcli -t --escape no -g "$1" "${@:2}") || die 'cannot inspect NetworkManager ownership property'
    [[ $value != *$'\n'* && $value != *$'\r'* ]] || die 'NetworkManager scalar output is ambiguous'
    [[ $value != -- ]] || value=
    printf '%s' "$value"
}
while IFS= read -r name; do
    managed=$(nm_value GENERAL.NM-MANAGED device show "$name")
    state=$(nm_value GENERAL.STATE device show "$name")
    path=$(nm_value GENERAL.CON-PATH device show "$name")
    [[ $managed == yes && $state =~ ^(20|30)([[:space:]]|$) && -z $path ]] || die 'NetworkManager already has an active fabric connection, or device inspection is uncertain'
done <<< "$targets"
profiles=$(inspect nmcli -t --escape no -f UUID,FILENAME connection show)
loaded_files=; uuids=; seen_uuids=
# One paired snapshot prevents a native file from being accepted using one
# query while its UUID is absent from another query and never inspected.
while IFS= read -r record; do
    if [[ -z $record ]]; then
        [[ -z $profiles ]] || die 'NetworkManager profile table contains an ambiguous empty row'
        continue
    fi
    [[ $record == *:* && $record != *$'\r'* ]] || die 'cannot interpret paired NetworkManager profile metadata'
    uuid=${record%%:*}; filename=${record#*:}
    [[ $uuid =~ ^[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$ && $seen_uuids != *" $uuid "* ]] || die 'cannot enumerate NetworkManager profiles'
    seen_uuids+=" $uuid "; uuids+="$uuid"$'\n'
    if [[ -n $filename && $filename != -- ]]; then
        [[ $filename =~ ^/[A-Za-z0-9_./@:+\ -]+$ && $filename != */../* && $filename != */./* && $filename != */ && $filename != *//* ]] || die 'NetworkManager profile filename is outside the safe line-oriented subset'
        loaded_files+="$filename"$'\n'
    fi
done <<< "$profiles"
loaded_files=${loaded_files%$'\n'}; uuids=${uuids%$'\n'}
for directory in /etc/NetworkManager/system-connections /run/NetworkManager/system-connections /usr/lib/NetworkManager/system-connections; do
    [[ ! -L $directory ]] || die 'native NetworkManager profile directory is indirect'
    [[ -e $directory ]] || continue
    [[ -d $directory && -r $directory && -x $directory ]] || die 'cannot enumerate native NetworkManager profiles'
    ls -A -- "$directory" >/dev/null 2>&1 || die 'cannot list native NetworkManager profiles safely'
    for filename in "$directory"/* "$directory"/.[!.]* "$directory"/..?*; do
        [[ -e $filename || -L $filename ]] || continue
        [[ -f $filename && ! -L $filename && $filename != *$'\n'* && $filename != *$'\r'* ]] || die 'native NetworkManager profile metadata is uncertain'
        awk -v filename="$filename" '$0==filename { found=1 } END { exit !found }' <<< "$loaded_files" || die 'a native NetworkManager file is not loaded; ownership cannot be proven'
    done
done
while IFS= read -r uuid; do
    [[ -n $uuid ]] || continue
    kind=$(nm_value connection.type connection show uuid "$uuid")
    case $kind in
        802-11-wireless|wifi|gsm|cdma|bluetooth|loopback) continue ;;
        802-3-ethernet|ethernet|bridge|bond|team|vlan|macvlan|ipvlan|ovs-interface|ovs-port|ovs-bridge|veth) ;;
        *) die 'unsupported NetworkManager profile type; ownership is uncertain' ;;
    esac
    binding=$(nm_value connection.interface-name connection show uuid "$uuid")
    selector=$(nm_value match.interface-name connection show uuid "$uuid")
    driver=$(nm_value match.driver connection show uuid "$uuid")
    match_path=$(nm_value match.path connection show uuid "$uuid")
    kernel_match=$(nm_value match.kernel-command-line connection show uuid "$uuid")
    candidates=$targets
    for exact_name in "$binding" "$selector"; do
        [[ -n $exact_name ]] || continue
        [[ $exact_name =~ ^[A-Za-z0-9_.-]+$ ]] || die 'NetworkManager interface selector needs structured matching; leave the fabric unchanged'
        candidates=$(awk -v name="$exact_name" '$0==name' <<< "$candidates")
    done
    if [[ $kind == 802-3-ethernet || $kind == ethernet ]]; then
        mac=$(nm_value 802-3-ethernet.mac-address connection show uuid "$uuid")
        if [[ -n $mac ]]; then
            [[ $mac =~ ^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$ ]] || die 'cannot interpret NetworkManager MAC binding'
            mac=$(tr '[:upper:]' '[:lower:]' <<< "$mac")
            remaining=
            while IFS= read -r name; do
                [[ -n $name ]] || continue
                current_mac=$(awk -v name="$name" '$1==name { print $3 }' <<< "$links")
                permanent=$(inspect ethtool -P "$name")
                [[ $permanent =~ ^Permanent\ address:\ (([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2})$ && ${BASH_REMATCH[1]} != 00:00:00:00:00:00 ]] || die 'cannot determine permanent fabric MAC ownership'
                permanent_mac=$(tr '[:upper:]' '[:lower:]' <<< "${BASH_REMATCH[1]}")
                if [[ $mac == "$current_mac" || $mac == "$permanent_mac" ]]; then remaining+="$name"$'\n'; fi
            done <<< "$candidates"
            candidates=${remaining%$'\n'}
        fi
    fi
    [[ -z $candidates || ( -z $driver && -z $match_path && -z $kernel_match ) ]] || die 'NetworkManager has unresolved driver, path or command-line ownership; leave the fabric unchanged'
    case $kind in
        vlan|macvlan|ipvlan)
            parent=$(nm_value "$kind.parent" connection show uuid "$uuid")
            [[ -z $parent || $parent =~ ^[A-Za-z0-9_.-]+$ ]] || die 'cannot safely interpret NetworkManager parent ownership'
            if [[ -n $parent ]]; then
                name_exists "$parent" || die 'NetworkManager parent is not a known live interface; ownership is uncertain'
                if related_name "$parent"; then die 'NetworkManager profile has a fabric parent interface'; fi
            fi ;;
    esac
    [[ -z $candidates ]] || die 'a saved NetworkManager profile can own a fabric interface'
done <<< "$uuids"
echo 'PASS live fabric, effective netplan and NetworkManager ownership are clear'
