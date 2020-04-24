#!/bin/bash

SOURCE="https://obj.umiacs.umd.edu/deploy-core3d/software/msi_to_rgb/models/msi_to_smask/trn_iter_50000.caffemodel"
DEST="./models/msi_to_smask"

echo -e "Downloading pre-trained model from \n${SOURCE} \nto ${DEST}\n"
wget -N ${SOURCE} -P ${DEST}
