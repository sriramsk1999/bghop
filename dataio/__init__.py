def get_data(args, return_val=False, val_downscale=4.0, **overwrite_cfgs):
    dataset_type = args.data.get("type", "HOI")
    cfgs = {
        # "scale_radius": args.data.get("scale_radius", -1),
        "downscale": args.data.downscale,
        "data_dir": args.data.data_dir,
        "train_cameras": False,
        'len_data': args.data.len,
        'suf': args.suf,
        'enable_bimanual': args.enable_bimanual,
    }

    if dataset_type == "HOI":
        # for HO3D
        from .hoi import SceneDataset
    else:
        raise NotImplementedError

    cfgs.update(overwrite_cfgs)
    dataset = SceneDataset(**cfgs)
    if return_val:
        cfgs["downscale"] = val_downscale
        val_dataset = SceneDataset(**cfgs)
        return dataset, val_dataset
    else:
        return dataset
