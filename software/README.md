
# Usage (using pre-built docker images on [dockerhub](https://hub.docker.com/u/venkai))

Each module has its own docker image that is built using [NVcaffe-v17](https://github.com/venkai/caffe/tree/venkai_nvcaffe17_cuda10), cuda 10.1 and cudnn-v7 on ubuntu16.04:
- **[Shadow Removal](https://hub.docker.com/r/venkai/shadow-removal)** : `docker pull venkai/shadow-removal:latest`
- **[RGB inpainting](https://hub.docker.com/r/venkai/inpainting-rgb)** : `docker pull venkai/inpainting-rgb:latest`
- **[RGBD iterative inpainting](https://hub.docker.com/r/venkai/inpainting-iterative-rgbd)** : `docker pull venkai/inpainting-iterative-rgbd:latest`
- **[MSI2RGB](https://hub.docker.com/r/venkai/msi-to-rgb)**: `docker pull venkai/msi-to-rgb:latest`
- **[Joint segmentation/DHM-estimation](https://hub.docker.com/r/venkai/joint-seg-dhm)**: `docker pull venkai/joint-seg-dhm:latest`

To run any module, do
`docker run --gpus 1 -it venkai/[module-name]`
and follow the instructions in the prompt.

# Custom Build Instructions for various modules (optional)

A generic Dockerfile template is provided here to build any module.
Navigate to any module sub-folder and build the docker image using
`docker build -t [software-name] -f [path to Dockerfile.generic]`.

Example 1: navigate to `shadow_removal` and build using
`docker build -t venkai/shadow-removal -f ../Dockerfile.template`

Example 2: navigate to `inpainting/inpainting_rgb` and build using
`docker build -t venkai/inpainting-rgb -f ../../Dockerfile.template`

