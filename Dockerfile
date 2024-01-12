ARG BASE_IMAGE_TAG=latest
FROM ghcr.io/pyvista/pyvista:$BASE_IMAGE_TAG
RUN sudo apt-get update && sudo apt-get install -y build-essential
COPY . ${HOME}
WORKDIR ${HOME}
RUN pip install -r requirements.txt