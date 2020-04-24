# RGB Inpainting to Remove Occluding Trees in Satellite Imagery
![placeholder](https://obj.umiacs.umd.edu/deploy-core3d/software/inpainting/inpainting_rgb/placeholder.png)
Performs inpainting on RGB satellite imagery, attempting to remove occluding
trees and recover a photorealistic RGB footprint of the building underneath
the tree. There are 2 types of models:
* **RGBM** - This model aggressively removes trees at the expense of blurring
building textures.
* **RGBM-UI** - Removes trees conservatively, but preserves building textures.


# Usage
```
Running python inpaint_rgb.py without any arguments will perform inpainting
on the example input in this folder using the rgbm_ui model. Check the 
results folder to see if the output matches the expected output.


usage: inpaint_rgb.py [-h] [--gpu GPU] [--model-type MODEL_TYPE]
                      [--input INPUT] [--outputdir OUTPUTDIR]
                      [--outputfile OUTPUTFILE] [--fp16] [--trn-dir TRN_DIR]
                      [--iter ITER] [--extra-pad EXTRA_PAD]

Command for running the Inpainting Pipeline on RGB satellite imagery

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             the gpu that will be used, e.g "0"
  --model-type MODEL_TYPE
                        Model Type: either "rgbm" or "rgbm_ui"
  --input INPUT         path to the 3-channel RGB input file.
  --outputdir OUTPUTDIR
                        path to write output prediction
  --outputfile OUTPUTFILE
                        name of inpainted output RGB
  --fp16                whether to use FP16 inference.
  --trn-dir TRN_DIR     directory which contains caffe model for inference
  --iter ITER           which iteration model to choose (def: 50000)
  --extra-pad EXTRA_PAD
                        add extra mirror padding to input [sometimes improves
                        results at border pixels] (def: 0)

```

