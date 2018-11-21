wget https://www.dropbox.com/s/x5kcq8gjil4kj2p/000030-1001-1416-resume3e-4-5000_model_multi_iter2000.npz
mv *npz* 000030-1001-1416-resume3e-4-5000_model_multi_iter2000.npz
mv *npz* nontarget_attack

if [ ! -f ens_adv_inception_resnet_v2.ckpt.meta ]; then
    wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    tar -xvf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
    rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz
fi
mv *ckpt* nontarget_attack

docker pull iwiwi/nips17-adversarial


