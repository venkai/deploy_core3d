FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04 AS nvcaffe
LABEL maintainer="venkai139@gmail.com"

# Steps required to add any custom APT repository
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
       apt-get install -y --no-install-recommends \
       apt-transport-https \
       ca-certificates \
       gnupg \
       software-properties-common \
       wget

# Add Kitware APT repository for latest cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - && \
        apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main' && apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install kitware-archive-keyring && \
        apt-key --keyring /etc/apt/trusted.gpg del C1F34CDD40CD72DA

# Install Libraries needed for NVcaffe
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
        apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        git \
        libopenblas-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        libpython2.7-dev \
        python-numpy \
        python-pip \
        python-scipy \
        python-protobuf \
        libjpeg-turbo8-dev libjpeg8-dev libturbojpeg && \
        rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# Clone NVcaffe from github repo.
ENV VCS_URL=https://github.com/venkai/caffe.git
ENV VCS_REF=venkai_nvcaffe17_cuda10
RUN git clone -b ${VCS_REF} --single-branch --depth 1 ${VCS_URL} .

# Install python dependencies first.
RUN pip2 install -U setuptools && \
        pip2 install -U wheel && \
        for req in $(cat python/requirements.txt) pydot; do pip2 install $req; done

# Build NVcaffe and python interface.
FROM nvcaffe
RUN mkdir build && cd build && \
        cmake ${CMAKE_ARGS} .. && \
        make -j"$(nproc)" && make pycaffe

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace

CMD ["/bin/bash"]

LABEL org.label-schema.vendor="UMIACS" \
      org.label-schema.name="venkai/nvcaffe" \
      org.label-schema.build-date="2020-04-22T07:31:12Z" \
      org.label-schema.vcs-url=$VCS_URL \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.schema-version="17.0" \
      com.nvidia.volumes.needed="nvidia_driver" \
      com.nvidia.cuda.version="${CUDA_VERSION}" \
      com.nvidia.cudnn.version="${CUDNN_VERSION}"
