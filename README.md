# Modules for Shadow-Removal, Inpainting and MSI2RGB
![placeholder](https://obj.umiacs.umd.edu/deploy-core3d/placeholder.png)
The list of modules are as follows:
* Joint Shadow Removal and Shadow Probability Estimation for Satellite Imagery.
* Inpainting
    1. RGB to RGB inpainting
    2. RGBD to RGBD tiled iterative inpainting.
* 8-band MSI (unknown normalization/sensor data) to photo-realistic RGB, shadow-free RGB and shadow probabilities. 

# Usage (using pre-built docker images on [dockerhub](https://hub.docker.com/u/venkai))

Each module has its own docker image that is built using [NVcaffe-v17](https://github.com/venkai/caffe/tree/venkai_nvcaffe17_cuda10), cuda 10.1 and cudnn-v7 on ubuntu16.04:
- **[Shadow Removal](https://hub.docker.com/r/venkai/shadow-removal)** : `docker pull venkai/shadow-removal:latest`
- **[RGB inpainting](https://hub.docker.com/r/venkai/inpainting-rgb)** : `docker pull venkai/inpainting-rgb:latest`
- **[RGBD iterative inpainting](https://hub.docker.com/r/venkai/inpainting-iterative-rgbd)** : `docker pull venkai/inpainting-iterative-rgbd:latest`
- **[MSI2RGB](https://hub.docker.com/r/venkai/msi-to-rgb)**: `docker pull venkai/msi-to-rgb:latest`

To run any module, do
`docker run --gpus 1 -it venkai/[module-name]`
and follow the instructions in the prompt.

Note that you need a CUDA capable GPU and NVIDIA Linux x86_64 Driver Version >= 418.39.

# Custom Build Instructions (optional)

If you wish to build NVCaffe from scratch, navigate to `./nvcaffe_docker` and build the Dockerfile using
`docker build -t venkai/nvcaffe .`
If you have an older NVIDIA driver that you don't want to upgrade, you can modify the first line of the
Dockerfile with the appropriate [nvidia/cuda image](https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md) using the compatibility table provided [here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver).

To build any module, a generic Dockerfile template is provided in `./software/Dockerfile.generic`.
Navigate to any module sub-folder and build the docker image using
`docker build -t [software-name] -f [path to Dockerfile.generic]`.
Example: navigate to `software/shadow_removal` and build using
`docker build -t venkai/shadow-removal -f ../Dockerfile.template`

# License & Citation
All modules are released under a variant of the [BSD 2-Clause license](https://github.com/venkai/deploy_core3d/blob/master/LICENSE). 

If you find any of our modules useful in your research, please consider citing our relevant papers:

```
@inproceedings{santhanam2017generalized,
    title={Generalized deep image to image regression},
    author={Santhanam, Venkataraman and Morariu, Vlad I and Davis, Larry S},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={5609--5619},
    year={2017}
}

@ARTICLE{8804390,
    author={V. {Santhanam} and L. S. {Davis}},
    journal={IEEE Transactions on Neural Networks and Learning Systems},
    title={A Generic Improvement to Deep Residual Networks Based on Gradient Flow},
    year={2019}, volume={}, number={}, pages={1-10},
}
```

# Acknowledgements
The research is based upon work supported by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon.


