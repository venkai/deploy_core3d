#!/bin/bash

ROOT="https://obj.umiacs.umd.edu/deploy-core3d/software/joint_seg_dhm"
for DEST in models/*/ ; do
    SOURCE="${ROOT}/${DEST}trn_iter_50000.caffemodel"
    echo -e "Downloading pre-trained model from \n${SOURCE} \nto ${DEST}\n"
    wget -N ${SOURCE} -P ${DEST}
done

