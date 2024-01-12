ARG BASE_IMAGE_TAG=latest
FROM ghcr.io/pyvista/pyvista:$BASE_IMAGE_TAG

USER root
RUN apt-get update && apt-get install -y build-essential
RUN apt-get install -y libopencv-dev
#RUN apt-get install -y libegl1

#~data folder was not writable (set to be writable only by `root`)
COPY . ${HOME}
RUN chown -R ${NB_USER}: /home/${NB_USER}

USER ${NB_USER}
WORKDIR ${HOME}
RUN pip install -r requirements.txt