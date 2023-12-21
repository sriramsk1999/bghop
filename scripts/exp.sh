# generation
python -m generate \
    S=5 cat_list=bowl+camera+hammer+binoculars+flashlight \


# train
CUDA_VISIBLE_DEVICES=1 python -m  ddpm3d.base -m \
    expname=dev/try2 \


# grasp synthesis
CUDA_VISIBLE_DEVICES=1  python  -m grasp_syn -m  \
    grasp_dir=/home/yufeiy2/scratch/result/GenGrasps/HO3D   \


# video recon
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m train  -m   \
    expname=hoi4d/\${data.index} \
    data.cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar data.ind=1,2 \
    environment.data_dir=/grogu/user/yufeiy2/gen3d_prerelease/ \
    +engine=grogu



