#!/bin/bash

SOURCE="https://obj.umiacs.umd.edu/deploy-core3d/software/inpainting/inpainting_rgb/models/rgbm/trn_iter_50000.caffemodel"
DEST="./models/rgbm"

echo -e "Downloading pre-trained model from \n${SOURCE} \nto ${DEST}\n"
wget -N ${SOURCE} -P ${DEST}

SOURCE="https://obj.umiacs.umd.edu/deploy-core3d/software/inpainting/inpainting_rgb/models/rgbm_ui/trn_iter_50000.caffemodel"
DEST="./models/rgbm_ui"

echo -e "Downloading pre-trained model from \n${SOURCE} \nto ${DEST}\n"
wget -N ${SOURCE} -P ${DEST}
