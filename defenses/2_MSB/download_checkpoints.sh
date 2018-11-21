#!/usr/bin/env bash

if [ ! -f adv_inception_v3.ckpt.meta ]; then
    wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
    tar -xvf adv_inception_v3_2017_08_18.tar.gz
    rm adv_inception_v3_2017_08_18.tar.gz
fi

if [ ! -f ens3_adv_inception_v3.ckpt.meta ]; then
    wget http://download.tensorflow.org/models/ens3_adv_inception_v3_2017_08_18.tar.gz
    tar -xvf ens3_adv_inception_v3_2017_08_18.tar.gz
    rm ens3_adv_inception_v3_2017_08_18.tar.gz
    mv 
fi

if [ ! -f ens4_adv_inception_v3.ckpt.meta ]; then
    wget http://download.tensorflow.org/models/ens4_adv_inception_v3_2017_08_18.tar.gz
    tar -xvf ens4_adv_inception_v3_2017_08_18.tar.gz
    rm ens4_adv_inception_v3_2017_08_18.tar.gz
fi

if [ ! -f ens_adv_inception_resnet_v2.ckpt.meta ]; then
    wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    tar -xvf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz
fi

if [ ! -f adv_inception_resnet_v2.ckpt.meta ]; then
    wget http://download.tensorflow.org/models/adv_inception_resnet_v2_2017_12_18.tar.gz
    tar -xvf adv_inception_resnet_v2_2017_12_18.tar.gz
    rm adv_inception_resnet_v2_2017_12_18.tar.gz
fi

mv *ckpt* defenses/2MSB



