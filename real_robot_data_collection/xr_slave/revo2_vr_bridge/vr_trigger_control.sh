#!/usr/bin/env bash
set -euo pipefail

hands="${REVO2_HANDS:-both}"
container_name="revo2-vr-trigger-${hands}-control"
feedback_name="pi05-revo2-feedback"
absolute_name="pi05-revo2-control"
image="zbl-registry.cn-shenzhen.cr.aliyuncs.com/xr/runtime/cx002:core_v00.28.02"
bridge_dir="/home/xr/robocontrol_ws/revo2_vr_bridge"
lock_file="/home/xr/robocontrol_ws/.gripper_control.lock"
left_port="${REVO2_LEFT_PORT:-/dev/serial/by-path/pci-0000:c7:00.3-usb-0:1.1.1:1.0-port0}"
right_port="${REVO2_RIGHT_PORT:-/dev/serial/by-path/pci-0000:c5:00.3-usb-0:3.1:1.0-port0}"
left_serial="${REVO2_LEFT_SERIAL:-}"
right_serial="${REVO2_RIGHT_SERIAL:-}"
grip_source_ip="${REVO2_GRIP_SOURCE_IP:-}"

exec 9>"${lock_file}"
if ! flock -n 9; then
    echo "ERROR: another gripper-control start/stop operation is already running" >&2
    exit 1
fi

verify_port() {
    local label="$1"
    local port="$2"
    local resolved
    resolved="$(realpath -e -- "${port}")" || {
        echo "ERROR: ${label} Revo2 serial port is missing: ${port}" >&2
        return 1
    }
    [[ -c "${resolved}" ]] || {
        echo "ERROR: ${label} Revo2 path is not a character device: ${resolved}" >&2
        return 1
    }
    echo "${label}=${port} -> ${resolved}"
}

wait_for_serial_release() {
    local attempt
    for attempt in {1..20}; do
        if ! fuser "$@" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.25
    done
    echo "ERROR: Revo2 serial port is still owned after 5 seconds" >&2
    fuser -v "$@" >&2 || true
    return 1
}

start_bridge() {
    [[ -n "${grip_source_ip}" ]] || {
        echo "ERROR: REVO2_GRIP_SOURCE_IP must identify the XR master" >&2
        return 2
    }
    case "${hands}" in
        left)
            [[ -n "${left_serial}" ]] || {
                echo "ERROR: REVO2_LEFT_SERIAL is required" >&2
                return 2
            }
            verify_port left "${left_port}"
            ;;
        right)
            [[ -n "${right_serial}" ]] || {
                echo "ERROR: REVO2_RIGHT_SERIAL is required" >&2
                return 2
            }
            verify_port right "${right_port}"
            ;;
        both)
            [[ -n "${left_serial}" && -n "${right_serial}" ]] || {
                echo "ERROR: REVO2_LEFT_SERIAL and REVO2_RIGHT_SERIAL are required" >&2
                return 2
            }
            verify_port left "${left_port}"
            verify_port right "${right_port}"
            [[ "$(realpath -e -- "${left_port}")" != "$(realpath -e -- "${right_port}")" ]] || {
                echo "ERROR: left and right hand ports resolve to the same device" >&2
                return 1
            }
            ;;
        *)
            echo "ERROR: REVO2_HANDS must be left, right, or both" >&2
            return 2
            ;;
    esac

    # Exactly one process may own each Revo2 Modbus serial device.
    docker stop -t 2 \
        "${feedback_name}" \
        "${absolute_name}" \
        revo2-vr-trigger-left-control \
        revo2-vr-trigger-right-control \
        revo2-vr-trigger-both-control \
        revo2-vr-trigger-control \
        revo2-vr-trigger-bridge \
        >/dev/null 2>&1 || true

    case "${hands}" in
        left)
            wait_for_serial_release "$(realpath -e -- "${left_port}")"
            ;;
        right)
            wait_for_serial_release "$(realpath -e -- "${right_port}")"
            ;;
        both)
            wait_for_serial_release \
                "$(realpath -e -- "${left_port}")" \
                "$(realpath -e -- "${right_port}")"
            ;;
    esac

    # Recreate so command-line safety and transport settings cannot remain stale.
    docker rm -f "${container_name}" >/dev/null 2>&1 || true
    docker run -d \
            --name "${container_name}" \
            --restart on-failure:3 \
            --network host \
            --privileged \
            --security-opt label=disable \
            -e PYTHONUNBUFFERED=1 \
            -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
            -e CYCLONEDDS_URI=file:///opt/xr/config/cyclone_uri/ser7.cyclonedds.xml \
            -e REVO2_HANDS="${hands}" \
            -e REVO2_LEFT_PORT="${left_port}" \
            -e REVO2_RIGHT_PORT="${right_port}" \
            -e REVO2_LEFT_SERIAL="${left_serial}" \
            -e REVO2_RIGHT_SERIAL="${right_serial}" \
            -e REVO2_GRIP_SOURCE_IP="${grip_source_ip}" \
            -v /opt/xr/config:/opt/xr/config \
            -v "${bridge_dir}":/bridge \
            -v /home/xr/.venvs/brainco-revo2:/opt/brainco-revo2 \
            -v /dev:/dev \
            "${image}" \
            /bin/bash -lc \
            'source /opt/ros/jazzy/setup.bash && export PYTHONPATH=/opt/brainco-revo2/lib/python3.12/site-packages:/bridge:${PYTHONPATH:-} && exec python3 /bridge/vr_trigger_revo2_bridge.py --hands "${REVO2_HANDS}" --left-port "${REVO2_LEFT_PORT}" --right-port "${REVO2_RIGHT_PORT}" --input-source trigger --command-rate 10 --duration-ms 150 --input-timeout 0.3 --min-trigger-delta 0.01 --low-deadzone 0.02 --high-deadzone 0.98 --grip-udp-port 39157 --grip-source-ip "${REVO2_GRIP_SOURCE_IP}"'

    sleep 3
    started_at="$(docker container inspect "${container_name}" --format '{{.State.StartedAt}}')"
    bridge_logs="$(docker logs --since "${started_at}" "${container_name}" 2>&1 || true)"
    identity_ready=false
    if [[ "${hands}" == "both" ]]; then
        if [[ "${bridge_logs}" == *"Verified left Revo2"* &&
              "${bridge_logs}" == *"Verified right Revo2"* ]]; then
            identity_ready=true
        fi
    elif [[ "${bridge_logs}" == *"Verified ${hands} Revo2"* ]]; then
        identity_ready=true
    fi
    if [[ "$(docker container inspect "${container_name}" --format '{{.State.Running}}' 2>/dev/null || true)" != "true" ]] ||
       [[ "${identity_ready}" != "true" ]]; then
        docker stop -t 1 "${container_name}" >/dev/null 2>&1 || true
        docker update --restart=no "${container_name}" >/dev/null 2>&1 || true
        docker logs --tail 80 "${container_name}" >&2 || true
        echo "ERROR: VR trigger bridge did not pass the ${hands}-hand readiness check" >&2
        return 1
    fi
    docker logs --tail 30 "${container_name}"
}

case "${1:-start}" in
    start)
        start_bridge
        ;;
    stop)
        docker stop -t 2 "${container_name}"
        ;;
    status)
        docker container inspect "${container_name}" \
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
