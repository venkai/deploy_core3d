
# Tiled Iterative RGBD Inpainting for Large Satellite Imagery

Performs tiled iterative rgbd inpainting that succesively shortens the height of
occluding trees, eventually recovering the RGBD footprint of occluded buildings.
Requires 3 inputs: RGB, DTM and DSM. (Only RGB + DHM is technically required.)
Produces 3 outputs: Inpainted RGB, DHM and DSM. (DTM remains the same.)

# Results on Synthetic Occlusions
![qual_synthetic](https://obj.umiacs.umd.edu/deploy-core3d/software/inpainting/inpainting_placeholders/qual.png)

# Results on Real Occlusions (No Ground Truth)

### Example 1
![r1](https://obj.umiacs.umd.edu/deploy-core3d/software/inpainting/inpainting_placeholders/r1.png)

### Example 2
![r2](https://obj.umiacs.umd.edu/deploy-core3d/software/inpainting/inpainting_placeholders/r2.png)

### Example 3
![r3](https://obj.umiacs.umd.edu/deploy-core3d/software/inpainting/inpainting_placeholders/r3.png)

### Example 4
![r4](https://obj.umiacs.umd.edu/deploy-core3d/software/inpainting/inpainting_placeholders/r4.png)

### Example 5
![r5](https://obj.umiacs.umd.edu/deploy-core3d/software/inpainting/inpainting_placeholders/r5.png)

# Usage

```
Run python inpaint_rgbd_tiled.py to perform tiled iterative inpainting with
9 repetitions on a large tile from Argentina (corresponding to the example inputs
within this folder.) Check the results folder to ensure that the outputs match
the expected_* outputs already in the results folder.

usage: inpaint_rgbd_tiled.py [-h] [--gpu GPU] [--nrep NREP]
                             [--input-rgb INPUT_RGB] [--input-dsm INPUT_DSM]
                             [--input-dtm INPUT_DTM] [--outputdir OUTPUTDIR]
                             [--outputfile OUTPUTFILE] [--fp16]
                             [--trn-dir TRN_DIR] [--iter ITER]
                             [--model-type MODEL_TYPE] [--extra-pad EXTRA_PAD]

Command for running the Inpainting Pipeline on RGBD satellite imagery

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             the gpu that will be used, e.g "0"
  --nrep NREP           repeated depth-inpainting iterations (def: 9)
  --input-rgb INPUT_RGB
                        path to the 3-channel RGB input file.
  --input-dsm INPUT_DSM
                        path to the DSM input file.
  --input-dtm INPUT_DTM
                        path to the DTM input file.
  --outputdir OUTPUTDIR
                        path to write output prediction
  --outputfile OUTPUTFILE
                        Inpainted output
  --fp16                whether to use FP16 inference.
  --trn-dir TRN_DIR     directory which contains caffe model for inference
  --iter ITER           which iteration model to choose (def: 0 [choose
                        latest])
  --model-type MODEL_TYPE
                        Model Type
  --extra-pad EXTRA_PAD
                        add extra mirror padding to input [sometimes improves
                        results at border pixels] (def: 0)

```


