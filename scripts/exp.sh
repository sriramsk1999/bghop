python -m ddpm3d.generate \
    S=5 cat_list=bowl+camera+hammer+binoculars+flashlight \


CUDA_VISIBLE_DEVICES=1 python -m  ddpm3d.base -m \
    expname=dev/try2 \


CUDA_VISIBLE_DEVICES=1  python  -m grasp_syn -m  \
    expname=sds_grasp/dev  \
    loss.w_hand=100     guide.w_obs=0 guide.w_sds=10  \
    sds_grasp=True    num=-1  \
    T=500  S=2  \
    index=null \
    grasp_dir=/home/yufeiy2/scratch/result/GenGrasps/HO3D   \

CUDA_VISIBLE_DEVICES=1  python  -m grasp_syn -m  \
    grasp_dir=/home/yufeiy2/scratch/result/GenGrasps/HO3D   \
