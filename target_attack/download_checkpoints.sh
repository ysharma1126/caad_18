
pip install tensorflow-gpu

if [ ! -f adv_inception_v3.ckpt.meta ]; then
    wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
    tar -xvf adv_inception_v3_2017_08_18.tar.gz
    rm adv_inception_v3_2017_08_18.tar.gz
    mkdir -p temp
    mv adv_inception_v3.ckpt* temp
    python rename_checkpoint.py \
        --checkpoint_dir=temp/adv_inception_v3.ckpt \
        --replace_from=InceptionV3 \
        --replace_to=AdvInceptionV3
    mv temp/* .
    rm -rf temp
fi

if [ ! -f ens3_adv_inception_v3.ckpt.meta ]; then
    wget http://download.tensorflow.org/models/ens3_adv_inception_v3_2017_08_18.tar.gz
    tar -xvf ens3_adv_inception_v3_2017_08_18.tar.gz
    rm ens3_adv_inception_v3_2017_08_18.tar.gz
    mkdir -p temp
    mv ens3_adv_inception_v3.ckpt* temp
    python rename_checkpoint.py \
        --checkpoint_dir=temp/ens3_adv_inception_v3.ckpt \
        --replace_from=InceptionV3 \
        --replace_to=Ens3AdvInceptionV3
    mv temp/* .
    rm -rf temp
fi

if [ ! -f ens4_adv_inception_v3.ckpt.meta ]; then
    wget http://download.tensorflow.org/models/ens4_adv_inception_v3_2017_08_18.tar.gz
    tar -xvf ens4_adv_inception_v3_2017_08_18.tar.gz
    rm ens4_adv_inception_v3_2017_08_18.tar.gz
    mkdir -p temp
    mv ens4_adv_inception_v3.ckpt* temp
    python rename_checkpoint.py \
        --checkpoint_dir=temp/ens4_adv_inception_v3.ckpt \
        --replace_from=InceptionV3 \
        --replace_to=Ens4AdvInceptionV3
    mv temp/* .
    rm -rf temp
fi

if [ ! -f ens_adv_inception_resnet_v2.ckpt.meta ]; then
    wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    tar -xvf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    mkdir -p temp
    mv ens_adv_inception_resnet_v2.ckpt* temp
    python rename_checkpoint.py \
        --checkpoint_dir=temp/ens_adv_inception_resnet_v2.ckpt \
        --replace_from=InceptionResnetV2 \
        --replace_to=EnsAdvInceptionResnetV2
    mv temp/* .
    rm -rf temp
fi

if [ ! -f adv_inception_resnet_v2.ckpt.meta ]; then
    wget http://download.tensorflow.org/models/adv_inception_resnet_v2_2017_12_18.tar.gz
    tar -xvf adv_inception_resnet_v2_2017_12_18.tar.gz
    rm adv_inception_resnet_v2_2017_12_18.tar.gz
    mkdir -p temp
    mv adv_inception_resnet_v2.ckpt* temp
    python rename_checkpoint.py \
        --checkpoint_dir=temp/adv_inception_resnet_v2.ckpt \
        --replace_from=InceptionResnetV2 \
        --replace_to=AdvInceptionResnetV2
    mv temp/* .
    rm -rf temp
fi

mv *ckpt* target_attack
