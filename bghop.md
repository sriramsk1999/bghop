# BGHOP

## Notes on G-HOP

- Changes:
  - Change wandb username in `ddpm3d/configs/environment/default.yaml` and `configs/environment/default.yaml`
  - Run dummy training with:
  `python -m ddpm3d.base`

- For some reason, there was some issue with the checkpoint having extra MANO params. Getting around it:
```
ckpt = torch.load("output/joint_3dprior/mix_data/checkpoints/last.ckpt")

for key in ckpt["state_dict"].keys():
   if 'verts_uv' in key:
      ckpt["state_dict"][key] = ckpt["state_dict"][key][:, :891, :]
   elif 'faces_uv' in key or 'hand_faces' in key:
      ckpt["state_dict"][key] = ckpt["state_dict"][key][:, :1538, :]

del ckpt["state_dict"]["glide_model.text_cond_model.model.text_model.embeddings.position_ids"]

torch.save("output/joint_3dprior/mix_data/checkpoints/last_modified.ckpt")
```

- Generate SDF for one arctic data point (remember to change paths):
```
python -m preprocess.make_sdf_grid --ds arctic_overfit
```
(I needed to set `PYOPENGL_PLATFORM=egl` to get it working)


- In `ddpm3d/dataset/arctic.py`:
  - Change paths for generated SDF and local ARCTIC dataset
  - Run with `python -m ddpm3d.base`

## Notes on generate.py

- Tracing the script:
```
  !!!torch.no_grad on this!!!
  sample, sample_list = model.forward
        get_model_kwargs - embeds text and creates nxyz if not created (stacks twice for conditional and unconditional)(from hoi3d_model.py)
        sample - diffusion steps
                samples.shape = torch.Size([3, 23, 16, 16, 16])
        decode
                obj is 3 channel grid - vqvae decodes to 64, 64, 64
                hand is 20 channel grid (however, paper says 15-channel grid)
  !!!torch.no_grad on this!!!
  model.decode_samples
        extract mesh from vqvae output
  model.hand_cond.grid2pose_sgd
        gradient descent to recover hand pose

  get nth transform -> transform normalized hand to actual hand coords
  concat meshes
```

## Notes on ARCTIC
- Git clone arctic dataset github repo
- We need to download only the following from ARCTIC:

```
export SMPLX_USERNAME=""
export SMPLX_PASSWORD=""
export MANO_USERNAME=""
export MANO_PASSWORD=""
./bash/download_body_models.sh # SMPLX and MANO
export ARCTIC_USERNAME=""
export ARCTIC_PASSWORD=""
./bash/download_misc.sh # raw_seqs/ (MANO and object poses in world coords) and meta/ (object meshes)

python scripts_data/unzip_download.py # unzip downloaded data
```

- Process the sequences with:
```
ERROR: Modify smplx package to return 21 joints for instead of 16
Solution: https://github.com/zc-alexfan/arctic/blob/master/docs/setup.md

python scripts_data/process_seqs.py --mano_p ./data/arctic_data/data/raw_seqs/s05/box_use_01.mano.npy --export_verts
```
This generates object and hand vertices in world coords. It saves a lot of other data and generates huge files, we should probably modify this to save exactly what we need. We really only need the object vertices in world coords, we can skip this and transform the object ourself but it'll need a little work since we also need to handle the object's articulation.

- Use the visualizer to view a sequence (need to download images also for this to work):
```
python scripts_data/visualizer.py --seq_p ./outputs/processed_verts/seqs/s05/box_use_01.npy --object --mano
```
