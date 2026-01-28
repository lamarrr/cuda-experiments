#!/bin/bash
set -euo pipefail

CONTAINER_VERSION="26.02"
CUDA_VERSION="13.0"
CONTAINER_NAME="ecstatic"

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
            --mount type=bind,src=$HOME/cuda-experiments,dst=/home/coder/cuda-experiments,consistency=consistent \
            --mount type=bind,src=$HOME/.config,dst=/home/coder/.config,consistency=consistent \
            --gpus=all \
            --name "${CONTAINER_NAME}" \
            -d "${CONTAINER_TAG}" /bin/bash
    fi

    docker attach "${CONTAINER_NAME}"
}

build() {
    mkdir -p build
    cd build
    cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="native"
    ninja
}

usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  launch-container  Create and attach to the Docker container"
    echo "  build             Build the project with CMake and Ninja"
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
    *)
        echo "Error: Unknown command '$1'"
        usage
        ;;
esac
