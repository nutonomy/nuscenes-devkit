FROM continuumio/miniconda3:4.6.14
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        xvfb && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /nuscenes-dev
# create conda nuscenes env
ARG PYTHON_VERSION
RUN bash -c "conda create -y -n nuscenes python=${PYTHON_VERSION} \
    && source activate nuscenes \
    && conda clean --yes --all"

COPY setup/requirements.txt .
COPY setup/requirements/ requirements/
# Install Python dependencies inside of the Docker image via pip & Conda.
# pycocotools installed from conda-forge
RUN bash -c "source activate nuscenes \
    && find . -name "\\*.txt" -exec sed -i -e '/pycocotools/d' {} \; \
    && pip install --no-cache -r /nuscenes-dev/requirements.txt \
    && conda config --append channels conda-forge \
    && conda install --yes pycocotools \
    && conda clean --yes --all"