#!/usr/bin/env bash
set -euo pipefail

container_name="revo2-vr-grip-sender"
bridge_dir="/home/xr/robocontrol_ws/revo2_vr_bridge"
destination="${REVO2_GRIP_DESTINATION:-}"
port="${REVO2_GRIP_PORT:-39157}"
image="${REVO2_GRIP_IMAGE:-zbl-registry.cn-shenzhen.cr.aliyuncs.com/xr/runtime/cx002:master_v00.28.01}"

start_sender() {
    [[ -n "${destination}" ]] || {
        echo "ERROR: REVO2_GRIP_DESTINATION must identify the XR slave" >&2
        return 2
    }
    docker rm -f "${container_name}" >/dev/null 2>&1 || true
    docker run -d \
        --name "${container_name}" \
        --restart unless-stopped \
        --network host \
        -e PYTHONUNBUFFERED=1 \
        -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
        -e CYCLONEDDS_URI=/opt/xr/config/cyclone_uri/local.cyclonedds.xml \
        -e REVO2_GRIP_DESTINATION="${destination}" \
        -e REVO2_GRIP_PORT="${port}" \
        -v /opt/xr/config:/opt/xr/config:ro \
        -v "${bridge_dir}":/bridge:ro \
        "${image}" \
        bash -lc \
        'source /opt/xr/bot/setup.bash && exec python3 /bridge/vr_grip_udp_sender.py --destination "${REVO2_GRIP_DESTINATION}" --port "${REVO2_GRIP_PORT}" --hands both --suppress-side-grip-pause --publish-no-pause-inputs'

    sleep 2
    if [[ "$(docker inspect "${container_name}" --format '{{.State.Running}}')" != "true" ]]; then
        docker logs --tail 80 "${container_name}" >&2 || true
        return 1
    fi
    docker logs --tail 20 "${container_name}"
}

case "${1:-start}" in
    start)
        docker run --rm --entrypoint /bin/true "${image}" >/dev/null
        start_sender
        ;;
    stop)
        docker stop -t 2 "${container_name}"
        ;;
    status)
        docker inspect "${container_name}" \
            --format 'status={{.State.Status}} restart={{.HostConfig.RestartPolicy.Name}}'
        ;;
    logs)
        docker logs --tail 100 "${container_name}"
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs}" >&2
        exit 2
        ;;
esac
