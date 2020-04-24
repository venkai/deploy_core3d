# Joint Shadow Removal and Shadow Probability Estimation

Takes an RGB satellite image containing shadows and produces a shadow-free
RGB image as well as a shadow probability map.

# Motivation
![motivation](https://obj.umiacs.umd.edu/deploy-core3d/software/shadow_removal/shadow_placeholders/motivation.png)

# Training
## [ISTD dataset](https://drive.google.com/file/d/1I0qw-65KBA6np8vIZzO6oeiOvcDBttAY/view)
![istd](https://obj.umiacs.umd.edu/deploy-core3d/software/shadow_removal/shadow_placeholders/istd_dataset.png)

## [Network Architecture](https://github.com/venkai/RBDN)
![v1](https://obj.umiacs.umd.edu/deploy-core3d/software/shadow_removal/shadow_placeholders/rbdn_shadow.png)

## Joint Shadow Removal and Shadow Probability Estimation
![joint](https://obj.umiacs.umd.edu/deploy-core3d/software/shadow_removal/shadow_placeholders/rbdn_shadow_v2.png)

# Results
![r1](https://obj.umiacs.umd.edu/deploy-core3d/software/shadow_removal/shadow_placeholders/sr_v2_i1.png)
![r2](https://obj.umiacs.umd.edu/deploy-core3d/software/shadow_removal/shadow_placeholders/sr_v2_i2.png)
![r3](https://obj.umiacs.umd.edu/deploy-core3d/software/shadow_removal/shadow_placeholders/sr_v2_i3.png)
![r4](https://obj.umiacs.umd.edu/deploy-core3d/software/shadow_removal/shadow_placeholders/sr_v2_i4.png)

# Usage
```
Run python remove_shadow.py without any arguments to perform shadow removal
on the example image in this folder. Check the results folder to see if the
outputs match the expected_* outputs already provided in the results folder.

usage: remove_shadow.py [-h] [--gpu GPU] [--input INPUT]
                        [--outputdir OUTPUTDIR]
                        [--outputrgbfile OUTPUTRGBFILE]
                        [--outputconffile OUTPUTCONFFILE] [--fp16]
                        [--trn-dir TRN_DIR] [--iter ITER]

Command for running the Shadow Removal Pipeline

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             the gpu that will be used, e.g "0"
  --input INPUT         path to the 3-channel RGB input file.
  --outputdir OUTPUTDIR
                        path to write output prediction.
  --outputrgbfile OUTPUTRGBFILE
                        name of shadow-free output RGB
  --outputconffile OUTPUTCONFFILE
                        shadow confidence values (as logits)
  --fp16                whether to use FP16 inference.
  --trn-dir TRN_DIR     directory which contains caffe model for inference
  --iter ITER           which iteration model to choose (def: 45000)


```


