ARG BASE_IMAGE_TAG=latest
FROM ghcr.io/pyvista/pyvista:$BASE_IMAGE_TAG

USER root
RUN apt-get update && apt-get install -y build-essential
RUN apt-get install -y libopencv-dev
#RUN apt-get install -y libegl1

USER ${NB_USER}
COPY . ${HOME}
WORKDIR ${HOME}
RUN pip install -r requirements.txt