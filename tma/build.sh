docker run --gpus=all --name ecstatic-monk -it -d rapidsai/devcontainers:26.02-cpp-cuda13.0-ubuntu24.04
docker attach ecstatic-monk 
sudo apt update && sudo apt upgrade

