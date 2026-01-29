#!/bin/bash
set -euo pipefail

CONTAINER_VERSION="26.02"
CUDA_VERSION="13.0"
CONTAINER_NAME="ecstatic"
WORKSPACE_DIR="$HOME/cuda-experiments"

launch_container() {
    CONTAINER_TAG="rapidsai/devcontainers:${CONTAINER_VERSION}-cpp-cuda${CUDA_VERSION}-ubuntu24.04"
    
    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container '${CONTAINER_NAME}' already exists. Attaching..."
        docker start "${CONTAINER_NAME}" 2>/dev/null || true
    else
        echo "Creating container '${CONTAINER_NAME}'..."
        docker run -it \
        --privileged \
        --mount type=bind,src=$HOME/.bash_history,dst=/home/.bash_history,consistency=consistent \
        --mount type=bind,src=$WORKSPACE_DIR,dst=/home/coder/cuda-experiments,consistency=consistent \
        --mount type=bind,src=$HOME/.config,dst=/home/coder/.config,consistency=consistent \
        --gpus=all \
        --name "${CONTAINER_NAME}" \
        -d "${CONTAINER_TAG}" \
        /bin/bash
    fi
    
    docker attach "${CONTAINER_NAME}"
}

build() {
    mkdir -p build
    cd build
    cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="native"
    ninja
}


nsys_profile(){
    OUTPUT=$1
    ARGS=$2
    nsys profile \
    -t nvtx,cuda,osrt \
    -f true \
    --cuda-memory-usage=true \
    --cuda-um-cpu-page-faults=true \
    --cuda-um-gpu-page-faults=true \
    --gpu-metrics-device=0 \
    --output="$OUTPUT" \
    $ARGS
}


ncu_profile(){
    OUTPUT=$1
    ARGS=$2
    SOURCE_FOLDER=$WORKSPACE_DIR/tma
    ncu \
    --config-file off \
    --export "$OUTPUT" \
    --force-overwrite \
    --set full \
    --call-stack \
    --call-stack-type native \
    --call-stack-type python \
    --nvtx \
    --import-source yes \
    --print-summary per-kernel \
    --source-folder $SOURCE_FOLDER \
    $ARGS
}


benchmark(){
    ARGS=$1
    
    ./$WORKSPACE_DIR/tma/build/tma_benchmark \
    -d 0 \
    --json "$WORKSPACE_DIR/tma/benchmark.json" \
    --markdown "$WORKSPACE_DIR/tma/benchmark.md" \
    --csv "$WORKSPACE_DIR/tma/benchmark.csv" \
    $ARGS
}


profile(){
    nsys_profile  "$WORKSPACE_DIR/tma_nsys_report.nsys-rep" "$WORKSPACE_DIR/tma/build/tma_benchmark -d 0 --run-once"
    ncu_profile  "$WORKSPACE_DIR/tma_ncu_report.ncu-rep" "$WORKSPACE_DIR/tma/build/tma_benchmark -d 0 --run-once"
}

usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  launch-container  Create and attach to the Docker container"
    echo "  build             Build the project with CMake and Ninja"
    echo "  benchmark         Run the benchmark"
    echo "  profile           Profile the benchmark with Nsight Systems and Nsight Compute"
    exit 1
}

# Require exactly one argument
if [[ $# -ne 1 ]]; then
    usage
fi

case "$1" in
    launch-container)
        launch_container
    ;;
    build)
        build
    ;;
    benchmark)
        benchmark
    ;;
    profile)
        profile
    ;;
    *)
        echo "Error: Unknown command '$1'"
        usage
    ;;
esac
