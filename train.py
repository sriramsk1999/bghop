# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import functools
import logging
import os
import os.path as osp
import sys
import time
from copy import deepcopy

import torch
import torch.distributed as dist
from hydra import main
from jutils import mesh_utils, slurm_utils
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import tools.vis_clips as tool_clip
from dataio import get_data
from models.base import get_optimizer, get_scheduler
from models.cameras import get_camera
from models.frameworks import get_model
from utils import io_util, mesh_util, rend_util, train_util
from utils.checkpoints import CheckpointIO
from utils.dist_util import (
    get_local_rank,
    get_rank,
    get_world_size,
    init_env,
    is_master,
)
from utils.logger import Logger
from utils.print_fn import log


def main_function(gpu=None, ngpus_per_node=None, args=None):
    init_env(args, gpu, ngpus_per_node)

    # ----------------------------
    # -------- shortcuts ---------
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    i_backup = int(args.schdl.i_backup // world_size) if args.schdl.i_backup > 0 else -1
    i_val = int(args.schdl.i_val // world_size) if args.schdl.i_val > 0 else -1
    special_i_val_mesh = [
        int(i // world_size)
        for i in [100, args.training.warmup, 1000, 2000, 5000, args.training.num_iters]
    ]
    exp_dir = args.exp_dir
    mesh_dir = os.path.join(exp_dir, "meshes")
    metric_dir = os.path.join(exp_dir, "metrics")
    os.makedirs(metric_dir, exist_ok=True)

    device = torch.device("cuda", local_rank)
    print("local ran", device)

    # logger
    logger = Logger(
        log_dir=exp_dir,
        img_dir=os.path.join(exp_dir, "imgs"),
        monitoring=args.logging,
        monitoring_dir=os.path.join(exp_dir, "events"),
        rank=rank,
        is_master=is_master(),
        multi_process_logging=(world_size > 1),
        cfg=args,
    )

    log.info("=> Experiments dir: {}".format(exp_dir))

    if is_master():
        # backup codes
        io_util.backup(os.path.join(exp_dir, "backup"))

        # save configs
        io_util.save_config(args, os.path.join(exp_dir, "config.yaml"))

    dataset, val_dataset = get_data(
        args, return_val=True, val_downscale=args.data.get("val_downscale", 4.0)
    )
    bs = args.data.get("batch_size", None)
    # save GT obj
    gt_oObj = val_dataset.oObj
    mesh_utils.dump_meshes([osp.join(mesh_dir, "oObj")], gt_oObj)

    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        pin_memory=args.data.get("pin_memory", False),
        collate_fn=mesh_utils.collate_meshes,
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=mesh_utils.collate_meshes,
    )

    # Create model
    posenet, focal_net = get_camera(
        args, datasize=len(dataset) + 1, H=dataset.H, W=dataset.W
    )
    (
        model,
        trainer,
        render_kwargs_train,
        render_kwargs_test,
    ) = get_model(args, data_size=len(dataset) + 1, cam_norm=dataset.max_cam_norm)
    trainer.train_dataloader = dataloader
    trainer.val_dataloader = valloader
    trainer.init_camera(posenet, focal_net)
    trainer.to(device)
    model.to(device)
    posenet.to(device)
    focal_net.to(device)

    log.info(model)
    log.info("=> Nerf params: " + str(train_util.count_trainable_parameters(model)))
    log.info(
        "=> Camera params: "
        + str(
            train_util.count_trainable_parameters(posenet)
            + train_util.count_trainable_parameters(focal_net)
        )
    )

    trainer.H = dataset.H
    trainer.W = dataset.W
    render_kwargs_train["H"] = dataset.H
    render_kwargs_train["W"] = dataset.W
    render_kwargs_test["H"] = val_dataset.H
    render_kwargs_test["W"] = val_dataset.W
    valH, valW = render_kwargs_test["H"], render_kwargs_test["W"]
    render_kwargs_surface = deepcopy(render_kwargs_test)
    render_kwargs_surface["H"] = render_kwargs_train["H"] // args.data.surface_downscale
    render_kwargs_surface["W"] = render_kwargs_train["W"] // args.data.surface_downscale
    render_kwargs_surface["rayschunk"] = args.data.surface_rayschunk

    # build optimizer
    optimizer = get_optimizer(args, model, posenet, focal_net)
    trainer.optimizer = optimizer

    # checkpoints
    checkpoint_io = CheckpointIO(
        checkpoint_dir=os.path.join(exp_dir, "ckpts"), allow_mkdir=is_master()
    )
    if world_size > 1:
        dist.barrier()
    # Register modules to checkpoint
    checkpoint_io.register_modules(
        posenet=posenet,
        focalnet=focal_net,
        model=model,
        optimizer=optimizer,
    )

    # Load checkpoints
    load_dict = checkpoint_io.load_file(
        args.training.ckpt_file,
        ignore_keys=args.training.ckpt_ignore_keys,
        only_use_keys=args.training.ckpt_only_use_keys,
        map_location=device,
    )

    logger.load_stats("stats.p")  # this will be used for plotting
    it = load_dict.get("global_step", 0)
    epoch_idx = load_dict.get("epoch_idx", 0)

    # pretrain if needed. must be after load state_dict, since needs 'is_pretrained' variable to be loaded.
    # ---------------------------------------------
    # -------- init perparation only done in master
    # ---------------------------------------------
    if is_master():
        pretrain_config = {"logger": logger}
        if "lr_pretrain" in args.training:
            pretrain_config["lr"] = args.training.lr_pretrain
            if model.implicit_surface.pretrain_hook(pretrain_config):
                checkpoint_io.save(
                    filename="latest.pt".format(it), global_step=it, epoch_idx=epoch_idx
                )

    # build scheduler
    scheduler = get_scheduler(args, optimizer, last_epoch=it - 1)
    trainer.scheduler = scheduler
    t0 = time.time()
    log.info(
        "=> Start training..., it={}, lr={}, in {}".format(
            it, optimizer.param_groups[0]["lr"], exp_dir
        )
    )
    end = (it >= args.training.num_iters)
    with tqdm(range(args.training.num_iters), disable=not is_master()) as pbar:
        if is_master():
            pbar.update(it)
        while it <= args.training.num_iters and not end:
            try:
                for indices, model_input, ground_truth in dataloader:
                    int_it = int(it // world_size)
                    # do a warm up for the first warm_hand iterations
                    # if int_it < args.training.warm_hand:
                    #     trainer.warm_hand(args, indices, model_input, ground_truth, render_kwargs_train, int_it)
                    #     continue
                    # -------------------
                    # validate
                    # -------------------
                    if is_master():
                        if (
                            i_val > 1
                            and int_it % i_val == 0
                            or int_it in special_i_val_mesh
                        ):
                            print("validation!!!!!")
                            with torch.no_grad():
                                (val_ind, val_in, val_gt) = next(iter(valloader))

                                trainer.eval()
                                val_ind = val_ind.to(device)
                                loss_extras = trainer(
                                    args, val_ind, val_in, val_gt, render_kwargs_test, 0
                                )

                                target_rgb = val_gt["rgb"].to(device)
                                target_mask = val_in["object_mask"].to(device)

                                ret = loss_extras["extras"]

                                to_img = functools.partial(
                                    rend_util.lin2img,
                                    H=render_kwargs_test["H"],
                                    W=render_kwargs_test["W"],
                                    batched=render_kwargs_test["batched"],
                                )
                                logger.add_imgs(to_img(target_rgb), "gt/gt_rgb", it)
                                logger.add_imgs(
                                    to_img(target_mask.unsqueeze(-1).float()),
                                    "gt/gt_mask",
                                    it,
                                )

                                trainer.val(
                                    logger,
                                    ret,
                                    to_img,
                                    it,
                                    render_kwargs_test,
                                    val_ind,
                                    val_in,
                                    val_gt,
                                )

                    # TODO: this validation step is very ugly...
                    # -------------------
                    # validate mesh
                    # -------------------
                    if is_master():
                        if (
                            i_val > 1
                            and int_it % i_val == 0
                            or int_it in special_i_val_mesh
                        ):
                            print("validating mesh!!!!!")
                            with torch.no_grad():
                                io_util.cond_mkdir(mesh_dir)
                                try:
                                    mesh_util.extract_mesh(
                                        model.implicit_surface,
                                        N=64,
                                        filepath=os.path.join(
                                            mesh_dir, "{:08d}.ply".format(it)
                                        ),
                                        volume_size=args.data.get("volume_size", 3.0),
                                        show_progress=is_master(),
                                    )
                                    logger.add_meshes(
                                        "obj",
                                        os.path.join("meshes", "{:08d}.ply".format(it)),
                                        it,
                                    )

                                    jObj = mesh_utils.load_mesh(
                                        os.path.join(mesh_dir, "{:08d}.ply".format(it))
                                    )
                                    jObj.textures = mesh_utils.pad_texture(
                                        jObj,
                                    )
                                except ValueError:
                                    log.warn("No surface extracted; pass")
                                    pass

                    # -------------------
                    # validate rendering
                    # -------------------
                    if (
                        is_master()
                        and i_val > 1
                        and int_it % i_val == 0
                        or int_it in special_i_val_mesh
                    ):
                        logging.info("vis tool_clip run_render")
                        one_time_loader = DataLoader(
                            dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=mesh_utils.collate_meshes,
                        )
                        try:
                            print("run clip!!!!!")
                            with torch.no_grad():
                                file_list = tool_clip.run(
                                    one_time_loader,
                                    trainer,
                                    os.path.join(logger.log_dir, "meshes"),
                                    "%08d" % it,
                                    224,
                                    224,
                                    N=256,
                                    volume_size=args.data.get("volume_size", 2.0),
                                )

                            for file in file_list:
                                name = os.path.basename(file)[9:-4]
                                logger.add_gif_files(file, "render/" + name, it)
                        except ValueError:
                            log.warn("No surface extracted; pass")
                            pass

                    if it >= args.training.num_iters:
                        end = True
                        break

                    # -------------------
                    # train
                    # -------------------
                    start_time = time.time()
                    trainer.train()
                    optimizer.zero_grad()

                    ret = trainer.forward(
                        args,
                        indices,
                        model_input,
                        ground_truth,
                        render_kwargs_train,
                        it,
                    )

                    losses = ret["losses"]
                    extras = ret["extras"]

                    # all but contact
                    for k, v in losses.items():
                        losses[k] = torch.mean(v)
                        print(k, losses[k].item())

                    losses["total"].backward()
                    if args.training.clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            trainer.parameters(), args.training.clip
                        )
                    # grad_norms = train_util.calc_grad_norm(model=model, posenet=posenet, focalnet=focal_net)
                    grad_norms = train_util.calc_grad_norm(
                        trainer=trainer,
                    )
                    optimizer.step()
                    scheduler.step(it)  # NOTE: important! when world_size is not 1

                    # -------------------
                    # logging
                    # -------------------
                    # done every i_save seconds
                    if (args.schdl.i_save > 1) and (
                        time.time() - t0 > args.schdl.i_save
                    ):
                        if is_master():
                            checkpoint_io.save(
                                filename="latest.pt",
                                global_step=it,
                                epoch_idx=epoch_idx,
                            )
                        # this will be used for plotting
                        logger.save_stats("stats.p")
                        t0 = time.time()

                    if is_master():
                        # ----------------------------------------------------------------------------
                        # ------------------- things only done in master -----------------------------
                        # ----------------------------------------------------------------------------
                        pbar.set_postfix(
                            lr=optimizer.param_groups[0]["lr"],
                            loss_total=losses["total"].item(),
                        )

                        if i_backup > 1 and int_it % i_backup == 0 and it > 1:
                            checkpoint_io.save(
                                filename="{:08d}.pt".format(it),
                                global_step=it,
                                epoch_idx=epoch_idx,
                            )

                    # ----------------------------------------------------------------------------
                    # ------------------- things done in every child process ---------------------------
                    # ----------------------------------------------------------------------------

                    # -------------------
                    # log grads and learning rate
                    for k, v in grad_norms.items():
                        logger.add("grad", k + f"_{it%2}", v, it)
                    logger.add(
                        "learning rates", "whole", optimizer.param_groups[0]["lr"], it
                    )

                    # -------------------
                    # log losses
                    for k, v in losses.items():
                        logger.add("losses", k, v.data.cpu().numpy().item(), it)
                        # print losses
                    if it % args.schdl.print_freq == 0 and is_master():
                        print(args.expname)
                        print("Iters [%04d] %f" % (it, losses["total"]))
                        for k, v in losses.items():
                            if k != "total":
                                print("\t %010s:%.4f" % (k, v.item()))

                    # logger.add('metric', 'hA', (extras['hA'].cpu() - ground_truth['hA']).abs().mean(), it)

                    # -------------------
                    # log extras
                    names = [
                        "radiance",
                        "alpha",
                        "implicit_surface",
                        "implicit_nablas_norm",
                        "sigma_out",
                        "radiance_out",
                    ]
                    for n in names:
                        p = "whole"
                        # key = "raw.{}".format(n)
                        key = n
                        if key in extras:
                            logger.add(
                                "extras_{}".format(n),
                                "{}.mean".format(p),
                                extras[key].mean().data.cpu().numpy().item(),
                                it,
                            )
                            logger.add(
                                "extras_{}".format(n),
                                "{}.min".format(p),
                                extras[key].min().data.cpu().numpy().item(),
                                it,
                            )
                            logger.add(
                                "extras_{}".format(n),
                                "{}.max".format(p),
                                extras[key].max().data.cpu().numpy().item(),
                                it,
                            )
                            logger.add(
                                "extras_{}".format(n),
                                "{}.norm".format(p),
                                extras[key].norm().data.cpu().numpy().item(),
                                it,
                            )
                    if "scalars" in extras:
                        for k, v in extras["scalars"].items():
                            logger.add("scalars", k, v.mean(), it)

                    # ---------------------
                    # end of one iteration
                    end_time = time.time()
                    log.debug(
                        "=> One iteration time is {:.2f}".format(end_time - start_time)
                    )

                    it += world_size
                    if is_master():
                        pbar.update(world_size)
                # ---------------------
                # end of one epoch
                epoch_idx += 1

            except KeyboardInterrupt:
                if is_master():
                    checkpoint_io.save(
                        filename="latest.pt".format(it),
                        global_step=it,
                        epoch_idx=epoch_idx,
                    )
                    # this will be used for plotting
                logger.save_stats("stats.p")
                sys.exit()

    if is_master():
        checkpoint_io.save(
            filename="final_{:08d}.pt".format(it), global_step=it, epoch_idx=epoch_idx
        )
        logger.save_stats("stats.p")


@main(config_path="configs", config_name="diffhoi3d_bimanual", version_base=None)
@slurm_utils.slurm_engine()
def main_cfg(args):
    print(args)
    main_function(None, None, args)


if __name__ == "__main__":
    main_cfg()
