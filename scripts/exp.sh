python -m generate \
    S=5 cat_list=bowl+camera+hammer+binoculars+flashlight \


CUDA_VISIBLE_DEVICES=1 python -m  ddpm3d.base -m \
    expname=dev/try2 \



CUDA_VISIBLE_DEVICES=1  python  -m grasp_syn -m  \
    grasp_dir=/home/yufeiy2/scratch/result/GenGrasps/HO3D   \



CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m scripts.prune_config  -m   \
    expname=hoi4d/\${data.index} \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1,2 \
    environment.output= \


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m train  -m   \
    expname=hoi4d/\${data.index} \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1,2 \
    environment.data_dir=/grogu/user/yufeiy2/gen3d_prerelease/ \
    +engine=grogu



CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m train  -m   \
    expname=dev_minimal/\${data.index}_no_login \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1,2 \
    environment=grogu_judy \


CUDA_VISIBLE_DEVICES=1 python -m  ddpm3d.base -m \
    expname=dev/joint \

    model.first_stage.ckpt_path=\${output}/ae_hand_field_thick/all_2k_x4_vq-f4_none_up64/checkpoints/last.ckpt \


CUDA_VISIBLE_DEVICES=1 python -m  scripts.prune_config -m \
    expname=joint_3dprior/mix_data \
    data@trainsets=[mix_data] cache_data=False balance_class=True \
    ckpt=\${output}/thicker_prior/joint/checkpoints/last.ckpt \
    lib=sem_attr_lib \




CUDA_VISIBLE_DEVICES=1 python -m  ddpm3d.base -m \
    expname=dev/try2 \

