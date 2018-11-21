#!/usr/bin/env bash
mkdir -p model_ckpts
if [ ! -f ens_adv_inception_resnet_v2.ckpt.meta ]; then
    wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    tar -xvf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz
fi

if [ ! -f nasnet-a_large.ckpt ]; then
    wget http://download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz
    tar -xvf nasnet-a_large_04_10_2017.tar.gz
    rm nasnet-a_large_04_10_2017.tar.gz
fi

mv *ckpt* defenses/Dropout/model_ckpts


