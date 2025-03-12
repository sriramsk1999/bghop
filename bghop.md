# BGHOP

## Notes on G-HOP

- Changes:
  - Change wandb username in `ddpm3d/configs/environment/default.yaml` and `configs/environment/default.yaml`
  - Run dummy training with:
  `python -m ddpm3d.base`


- Core model classes - `ddpm3d/base.py` and `ddpm3d/hoi3d_model.py`
- Training script - `ddpm3d/base.py`
- Input to model:
```
    hA - Hand Articulation (45,)
    nXyz - Grid points, used for calculating skeletal distance field (?)
    image - Latent of object as a 3D grid
```

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

- We need to download only the following from ARCTIC:

```
./bash/download_body_models.sh # SMPLX and MANO
./bash/download_misc.sh # raw_seqs/ (MANO and object poses in world coords) and meta/ (object meshes)

python scripts_data/unzip_download.py # unzip downloaded data
```

- Process the sequences with:
```
python scripts_data/process_seqs.py --mano_p ./data/arctic_data/data/raw_seqs/s05/box_use_01.mano.npy --export_verts
```
This generates object and hand vertices in world coords. It saves a lot of other data and generates huge files, we should probably modify this to save exactly what we need. We really only need the object vertices in world coords, we can skip this and transform the object ourself but it'll need a little work since we also need to handle the object's articulation.

- Use the visualizer to view a sequence (need to download images also for this to work):
```
python scripts_data/visualizer.py --seq_p ./outputs/processed_verts/seqs/s05/box_use_01.npy --object --mano
```
