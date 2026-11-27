#!/usr/bin/env bash
# File contents shared by previews and installation. No writes here.
render_content() {
    local name=$1 i
    case $name in
    40-cx7.yaml)
        printf 'network:\n  version: 2\n  ethernets:\n'
        for i in "${!NETDEVS[@]}"; do
            printf '    %s:\n      renderer: NetworkManager\n      dhcp4: false\n      dhcp6: false\n      mtu: 9000\n      addresses: [%s]\n' "${NETDEVS[$i]}" "${LOCAL_CIDRS[$i]}"
        done ;;
    hosts.fragment)
        echo '# BEGIN VADE SPARK FABRIC'
        for i in "${!PEER_IPS[@]}"; do printf '%s %s\n' "${PEER_IPS[$i]}" "${PEER_NAMES[$i]}"; done
        echo '# END VADE SPARK FABRIC' ;;
    99-vade-memlock.conf)
        printf '%s soft memlock unlimited\n%s hard memlock unlimited\n' "$SERVICE_USER" "$SERVICE_USER" ;;
    ssh-memlock.conf) printf '[Service]\nLimitMEMLOCK=infinity\n' ;;
    service-memlock.conf) printf '[Service]\nLimitMEMLOCK=infinity\nSupplementaryGroups=%s\n' "$RDMA_GROUP" ;;
    99-vade-rdma.rules)
        printf 'KERNEL=="uverbs*", GROUP="%s", MODE="0660"\nKERNEL=="rdma_cm", GROUP="%s", MODE="0660"\n' "$RDMA_GROUP" "$RDMA_GROUP" ;;
    zzz-vade-kho.cfg)
        cat <<'KHO'
# Vade mitigation for NVIDIA's 7.0.0-1019-nvidia RoCE advisory.
for vade_arg in ${GRUB_CMDLINE_LINUX:-} ${GRUB_CMDLINE_LINUX_DEFAULT:-}; do
  case "$vade_arg" in kho=off) ;; kho=*) echo 'Conflicting kho setting; review GRUB configuration' >&2; exit 1 ;; esac
done
case " ${GRUB_CMDLINE_LINUX:-} ${GRUB_CMDLINE_LINUX_DEFAULT:-} " in
  *" kho=off "*) ;;
  *) GRUB_CMDLINE_LINUX_DEFAULT="${GRUB_CMDLINE_LINUX_DEFAULT:-} kho=off" ;;
esac
unset vade_arg
KHO
        ;;
    *) die 'unknown rendered file' ;;
    esac
}
