# MSI to photorealistic RGB, shadow-free RGB and shadow probabilities

Takes an 8-band MSI (with potentially unknown normalization/sensor-parameters) as input
and produces the following 3 outputs:
1. **Photorealistic RGB** : An RGB image with good color accuracy regardless of MSI normalization.
2. **Shadow-Free RGB** : Removes shadows from the previous output.
3. **Shadow Probability Map** : Provides per-pixel probabilities for likelihood of shadow.

![placeholder](https://obj.umiacs.umd.edu/deploy-core3d/software/msi_to_rgb/placeholder.png)

This is all achieved using an end-to-end DCNN based on RBDN, that is trained on the [GRSS dataset](https://github.com/pubgeo/dfc2019/tree/master/data)
containing photorealistic RGB images with registered MSI data. The MSI data is stripped of existing
normalization by independently normalizing each of the 8 channels to have zero mean and unit standard
deviation. Furthermore, the histograms of each channel are stretched and squeezed by up to 25% in
order to account for variations across different satellite sensors. The shadow removal component is
achieved by leveraging the existing work on shadow removal for satellite imagery.

# Usage
Modify `params.py` and do `python run_model.py`. 

Modify `INPUT_DIR` in `params.py` to point to a folder containing MSI inputs.
Modify `OUTPUT_DIR` to point to where you want to save the RGB, shadow-free RGB and shadow-probability outputs.

Run `python run_model.py` without any arguments to process the example images in the `inputs_msi` folder.
Check the `results` folder to see if the outputs match the ones in the `results/expected` folder.

