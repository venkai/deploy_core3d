# Joint Shadow Removal and Shadow Probability Estimation

Takes an RGB satellite image containing shadows and produces a shadow-free
RGB image as well as a shadow probability map.

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


