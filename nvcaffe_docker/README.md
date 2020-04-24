
# Build Instructions for NVcaffe (optional)

Build the Dockerfile using
`docker build -t venkai/nvcaffe .`
If you have an older NVIDIA driver that you don't want to upgrade, you can modify the first line of the
Dockerfile with the appropriate [nvidia/cuda image](https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md) using the compatibility table provided [here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver).


