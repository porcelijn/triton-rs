#!/bin/bash
set -e
set -u
set -o pipefail

shutdown_tritonserver() {
    if [ -z "$PID" ]; then
        echo "No PID, skipping tritonserver shutdown"
        return
    fi
    for SIG in SIGINT SIGTERM SIGKILL; do
        echo "Sending $SIG signal to tritonserver"
        kill -s $SIG $PID
        for retry in 3 2 1; do
            kill -s 0 $PID 2>/dev/null || break
            echo "Waiting for tritonserver shutdown $retry"
            sleep 1
        done
    done
}

trap shutdown_tritonserver ERR

# Running

# start tritonserver in background
tritonserver --exit-on-error=true \
             --model-control-mode=explicit \
             --model-repository=/model-repository \
             --load-model=example-test \
             --load-model=passthrough \
             & PID=$!

# Busy wait for liveness
for retry in 4 3 2 1; do
    if curl --fail --silent localhost:8000/v2/health/live; then
        echo "Client probe: tritonserver is LIVE"
        break
    fi
    sleep 1
    echo "Retrying liveness $retry"
done

# Busy wait for readiness
for retry in 4 3 2 1; do
    if curl --fail --silent localhost:8000/v2/health/ready; then
        echo "Client probe: tritonserver is READY"
        break
    fi
    sleep 1
    echo "Retrying readiness $retry"
done

# Health check
curl --fail -vs localhost:8000/v2/health/ready

# Model check
for model in example-test passthrough; do
    if curl --fail -s localhost:8000/v2/models/${model}/ready; then
        echo "Model ${model} READY"
    else
        echo "Model ${model} FAILED"
        curl -s localhost:8000/v2/models/${model}
        false
    fi
done

# Functional check
echo '{"model_name":"example-test","model_version":"1","outputs":[{"name":"output","datatype":"FP32","shape":[1],"data":[123.0]}]}' | jq > expected
curl --fail -d '{"inputs":[{"name":"prompt","data":[123], "datatype":"FP32", "shape":[1]}]}' http://localhost:8000/v2/models/example-test/infer | jq > actual
diff expected actual
