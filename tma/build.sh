CONTAINER_VERSION="26.02"
CUDA_VERSION="13.0"

function create_docker_container(){
    CONTAINER_TAG="rapidsai/devcontainers:${CONTAINER_VERSION}-cpp-cuda${CUDA_VERSION}-ubuntu24.04"
    CONTAINER_NAME=$1
    docker run -it \
    --privileged \
    --mount type=bind,src=$HOME/.bash_history,dst=/home/.bash_history,consistency=consistent \
    --mount type=bind,src=$HOME/cuda-experiments,dst=/home/coder/cuda-experiments,consistency=consistent \
    --mount type=bind,src=$HOME/.config,dst=/home/coder/.config,consistency=consistent \
    --gpus=all \
    --name $CONTAINER_NAME \
    -d $CONTAINER_TAG /bin/bash
}

CONTAINER_NAME="ecstatic"

create_docker_container $CONTAINER_NAME
docker attach $CONTAINER_NAME


