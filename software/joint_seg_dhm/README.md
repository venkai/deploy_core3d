# Modules for Joint Satellite Segmentation/DHM estimation

![placeholder](https://obj.umiacs.umd.edu/deploy-core3d/software/joint_seg_dhm/placeholders/placeholder.png)

Takes an 8-band MSI or 3-channel RGB or both as input and produce either DHM estimate or
segmentation estimate (ground, building, tree, water, road) or both jointly. In case DHM is
already known, there is another model provided that takes all 3 modalities (RGB, MSI and DHM)
as input and outputs a segmentation estimate. Refer to the [GRSS data page](https://github.com/pubgeo/dfc2019/tree/master/data#classification-labels) for information on how semantic classes are labelled.

In total, the following **8** modules are provided:
1. `msi_to_agl`: Takes MSI as input and produces DHM estimate.
2. `msi_to_cls`: Takes MSI as input and produces Segmentation estimate.
3. `rgb_to_agl`: Takes RGB as input and produces DHM estimate.
4. `rgb_to_cls`: Takes RGB as input and produces Segmentation estimate.
5. `rgb_msi_to_agl`: Takes RGB and MSI as input and produces DHM estimate.
6. `rgb_msi_to_cls`: Takes RGB and MSI as input and produces Segmentation estimate.
7. `rgb_msi_agl_to_cls`: Takes RGB, MSI and DHM as input and produces Segmentation estimate.
8. `rgb_msi_to_agl_cls`: Takes RGB and MSI as input and jointly estimates DHM and Segmentation labels.

This is all achieved using an end-to-end DCNN based on [RBDN](https://github.com/venkai/RBDN), that is trained on the [GRSS dataset](https://github.com/pubgeo/dfc2019/tree/master/data).

# Results

### Example 1
![r1](https://obj.umiacs.umd.edu/deploy-core3d/software/joint_seg_dhm/placeholders/r1.png)

### Example 2
![r2](https://obj.umiacs.umd.edu/deploy-core3d/software/joint_seg_dhm/placeholders/r2.png)

### Example 3
![r3](https://obj.umiacs.umd.edu/deploy-core3d/software/joint_seg_dhm/placeholders/r3.png)


# Usage
Modify `params.py` and do `python run_model.py`. 

Modify `TEST_DIR` in `params.py` to point to a folder containing the desired inputs.
Modify `OUTPUT_DIR` to point to where you want to save the relevant outputs.
The default module is the joint-estimation model `rgb_msi_to_agl_cls`. Look at `run_model.py`
for examples on how to change this to any other module.

Run `python run_model.py` without any arguments to process the example images in the `inputs` folder,
using each of the **8** modules provided. Check the `results` folder to see if the outputs match the
ones in the `results/expected` folder.

