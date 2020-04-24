#!/bin/bash

SOURCE="https://obj.umiacs.umd.edu/deploy-core3d/software/inpainting/inpainting_iterative_rgbd/models/rgbd/trn_iter_162000.caffemodel"
DEST="./models/rgbd"

echo -e "Downloading pre-trained model from \n${SOURCE} \nto ${DEST}\n"
wget -N ${SOURCE} -P ${DEST}
