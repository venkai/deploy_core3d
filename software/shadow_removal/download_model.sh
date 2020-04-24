#!/bin/bash

SOURCE="https://obj.umiacs.umd.edu/deploy-core3d/software/shadow_removal/models/trn_iter_45000.caffemodel"
DEST="./models"

echo -e "Downloading pre-trained model from \n${SOURCE} \nto ${DEST}\n"
wget -N ${SOURCE} -P ${DEST}
