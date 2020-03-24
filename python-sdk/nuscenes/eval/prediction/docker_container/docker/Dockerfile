ARG FROM
FROM ${FROM}

MAINTAINER nutonomy.com

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      libsm6 \
      libxext6 \
      libxrender-dev \
      libgl1-mesa-glx \
      libglib2.0-0 \
      xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN curl -OL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH /opt/miniconda3/bin:$PATH

RUN conda update -n base -c defaults conda

RUN groupadd -g 1000 dev \
    && useradd -d /home/nuscenes -u 1000 -g 1000 -m -s /bin/bash dev

USER dev

WORKDIR /nuscenes-dev/prediction

ENV PYTHONPATH=/nuscenes-dev/python-sdk

COPY setup/requirements.txt .

RUN bash -c "conda create -y -n nuscenes python=3.7 \
    && source activate nuscenes \
    && pip install --no-cache-dir -r /nuscenes-dev/prediction/requirements.txt \
    && conda clean --yes --all"

VOLUME [ '/nuscenes-dev/python-sdk', '/nuscenes-dev/prediction', '/data/sets/nuscenes', '/nuscenes-dev/Documents' ]
