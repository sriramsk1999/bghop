# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------

import json
import numpy as np
from hydra import main
import hydra.utils as hydra_utils
import pickle
from glob import glob
import os.path as osp
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from jutils import (
    mesh_utils,
    image_utils,
    geom_utils,
    hand_utils,
    slurm_utils,
    web_utils,
)
from models.sd import SDLoss
from utils.obj2text import Obj2Text
from utils.io_util import load_sdf_grid
from utils.contact_util import compute_contact_loss

device = "cuda:0"


class UniGuide:
    def __init__(self) -> None:
        self.hand_wrapper = hand_utils.ManopthWrapper().to(device)
        self.sd = None
        self.text_template = {}

    def init_sds(self, args, device):
        sd = SDLoss(args.load_pt, **args.sd_para)
        sd.init_model(device)
        self.sd = sd
        lib_name = sd.model.cfg.get("lib", None)
        if lib_name is not None:
            lib_name = osp.join(hydra_utils.get_original_cwd(), f"docs/{lib_name}.json")
        self.text_template = Obj2Text(lib_name)
        return sd

    def vis_nSdf_hA(self, nSdf, hA, hand_wrapper, save_name):
        N = len(nSdf)
        nSdf_mesh = mesh_utils.batch_grid_to_meshes(nSdf, N, half_size=1.5)
        nTh = hand_utils.get_nTh(hA=hA, hand_wrapper=hand_wrapper)
        nHand, _ = hand_wrapper(nTh, hA)

        nHand.textures = mesh_utils.pad_texture(nHand, "blue")
        nSdf_mesh.textures = mesh_utils.pad_texture(nSdf_mesh, "yellow")
        nHoi = mesh_utils.join_scene([nSdf_mesh, nHand])
        image_list = mesh_utils.render_geom_rot_v2(
            nHoi,
        )
        image_utils.save_gif(image_list, save_name)
        return

    @torch.no_grad()
    def eval_grasp(self, sd: SDLoss, nSdf_pred, hA_pred, text, cfg, save_pref, S=5):
        loss_record = []
        batch = sd.model.set_inputs({"nSdf": nSdf_pred, "hA": hA_pred})
        nXyz = mesh_utils.create_sdf_grid(1, nSdf_pred.shape[-1], 1.5, device=device)
        batch["nXyz"] = nXyz
        text = self.text_template(text)
        batch["text"] = text
        max_t = 100
        for t in tqdm(range(0, max_t), desc="eval grasp"):
            loss_s = []
            for s in range(S):
                loss, rtn = sd.apply_sd(
                    batch,
                    text=batch["text"],
                    it=t,
                    multi_scale=5,
                    t=t,
                    w_schdl="dream",
                    grad_mode="noise",
                )
                loss_s.append(loss.item())
            loss_record.append(np.array(loss_s).mean())
        mean_loss = np.array(loss_record).mean()
        print(mean_loss)
        return mean_loss

    def sds_grasp(
        self,
        sd: SDLoss,
        uObj,
        text,
        nTu=None,
        vis_every_n=10,
        w_sds=10,
        T=500,
        cfg=None,
        opt_nTu=True,
        save_pref="",
    ):
        """
        Args:
            sd (SDLoss): _description_
            uObj (_type_): SDF grid in shape of (N, 1, 64, 64, 64)
            text (_type_): _description_
            nTu (_type_, optional): _description_. Defaults to None.
            vis_every_n (int, optional): _description_. Defaults to 10.
            w_sds (int, optional): _description_. Defaults to 10.
            T (int, optional): _description_. Defaults to 1000.
            cfg (_type_, optional): _description_. Defaults to None.
            save_pref (str, optional): _description_. Defaults to '/home/yufeiy2/scratch/result/uni_guide/sds_grasp'.
        """
        bs = len(uObj)
        hA = self.hand_wrapper.hand_mean.clone().repeat(bs, 1)
        if nTu is not None:
            nSdf_gt, _ = mesh_utils.transform_sdf_grid(uObj, nTu, N=64, lim=1.5)
            nSdf_gt = nSdf_gt.clone()
            nTu_gt = nTu.clone()
            nTu_rot_gt, nTu_tsl_gt, nTu_scale_gt = geom_utils.homo_to_rt(nTu_gt)
            nTu_rot_gt = geom_utils.matrix_to_rotation_6d(nTu_rot_gt)
        else:
            nSdf_gt = uObj.clone()
            nTu_scale_gt = torch.ones([bs, 3], device=device)
            nTu_rot_gt = (
                geom_utils.matrix_to_rotation_6d(torch.eye(3)[None])
                .to(device)
                .repeat(bs, 1)
            )
            nTu_tsl_gt = torch.zeros([bs, 3], device=device)
        hA_pred = nn.Parameter(hA.clone())
        lr = 1e-2 * bs
        if opt_nTu:
            nTu_rot = nn.Parameter(nTu_rot_gt)
            nTu_tsl = nn.Parameter(nTu_tsl_gt)
            optimizer = torch.optim.AdamW([nTu_rot, nTu_tsl, hA_pred], lr=lr)
        else:
            nTu_rot = nTu_rot_gt
            nTu_tsl = nTu_tsl_gt
            optimizer = torch.optim.AdamW([hA_pred], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T, eta_min=lr / 100
        )

        text = self.text_template(text)
        nXyz = mesh_utils.create_sdf_grid(1, uObj.shape[-1], 1.5, device=device)

        self.vis_nSdf_hA(nSdf_gt, hA, self.hand_wrapper, save_pref + "_init")

        for t in tqdm(range(T + 1)):
            optimizer.zero_grad()

            nTu_cur = geom_utils.rt_to_homo(
                geom_utils.rotation_6d_to_matrix(nTu_rot), nTu_tsl, nTu_scale_gt
            )
            nSdf_pred, _ = mesh_utils.transform_sdf_grid(
                uObj, nTu_cur, N=64, lim=1.5 / 1.5, extra=False
            )
            batch = sd.model.set_inputs({"nSdf": nSdf_pred, "hA": hA_pred})
            batch["nXyz"] = nXyz
            batch["text"] = text

            loss, rtn = sd.apply_sd(
                batch, text=batch["text"], it=t, **cfg.loss, multi_scale=5
            )
            sds_loss = w_sds * loss

            losses = {"sds_loss": sds_loss}
            loss = 0
            for k, v in losses.items():
                loss = loss + v
            loss.backward()

            optimizer.step()
            scheduler.step()

            if t % vis_every_n == 0:
                print(
                    f"[Global Step] {t:04d} [Loss] {loss.item():.4f} [SDS Loss] {sds_loss.item():.4f}"
                )
                self.vis_nSdf_hA(
                    nSdf_pred, hA_pred, self.hand_wrapper, save_pref + f"_t{t:04d}_pred"
                )

        nTu_cur = geom_utils.rt_to_homo(
            geom_utils.rotation_6d_to_matrix(nTu_rot), nTu_tsl, nTu_scale_gt
        )
        nSdf_pred, _ = mesh_utils.transform_sdf_grid(
            uObj, nTu_cur, N=64, lim=1.5 / 1.5, extra=False
        )
        self.vis_nSdf_hA(
            nSdf_pred, hA_pred, self.hand_wrapper, save_pref + f"_t{t:04d}_pred"
        )

        nTu_cur = geom_utils.rt_to_homo(
            geom_utils.rotation_6d_to_matrix(nTu_rot), nTu_tsl, nTu_scale_gt
        )
        pred_loss = self.eval_grasp(sd, nSdf_pred, hA_pred, text, cfg, save_pref)

        return nTu_cur, hA_pred, pred_loss

    def refine_grasp(
        self,
        oObj,
        hA_origin,
        nTu,
        oTu,
        vis_every_n=100,
        T=100,
        save_pref="",
        w_pen=1.0,
        w_miss=1.0,
        w_damp=0.1,
    ):
        nTh = hand_utils.get_nTh(hA=hA_origin, hand_wrapper=self.hand_wrapper)
        uTn = geom_utils.inverse_rt(mat=nTu, return_mat=True)
        oTh = oTu @ uTn @ nTh

        oTh_rot_ori, oTh_trsl_ori, oTh_scale = geom_utils.homo_to_rt(oTh)
        oTh_rot_ori = geom_utils.matrix_to_rotation_6d(oTh_rot_ori)
        oTh_so6d = nn.Parameter(oTh_rot_ori)
        oTh_trsl = nn.Parameter(oTh_trsl_ori)
        hA = nn.Parameter(hA_origin)

        oHand_origin, _ = self.hand_wrapper(oTh, hA_origin)

        optimizer = torch.optim.Adam([oTh_so6d, oTh_trsl, hA], lr=1e-3)
        for t in range(T + 1):
            optimizer.zero_grad()
            oTh = geom_utils.rt_to_homo(
                geom_utils.rotation_6d_to_matrix(oTh_so6d),
                oTh_trsl,
                oTh_scale,
            )
            oHand, _ = self.hand_wrapper(oTh, hA)

            missed_loss, penetr_loss, _, _ = compute_contact_loss(
                oHand.verts_padded() * 100,
                oHand.faces_padded(),
                oObj.verts_padded() * 100,
                oObj.faces_padded()[0],
                contact_zones="zones",
            )
            loss, losses = 0.0, {}
            damp = F.mse_loss(oHand.verts_padded(), oHand_origin.verts_padded())

            losses["damp"] = w_damp * damp
            losses["missed"] = w_miss * missed_loss
            losses["penetr"] = w_pen * penetr_loss

            for k, v in losses.items():
                loss = loss + v

            loss.backward()
            optimizer.step()
            if t % 10 == 0:
                print(
                    f'Step [{t:4d}] Loss: {loss.item():f} \t damp: {losses["damp"].item():f} miss: {losses["missed"].item():f}, penetr: {losses["penetr"].item():f}'
                )

            if t % vis_every_n == 0:
                # print loss
                oHand.textures = mesh_utils.pad_texture(oHand, "blue")
                oObj.textures = mesh_utils.pad_texture(oObj, "yellow")
                image_list = mesh_utils.render_geom_rot_v2(
                    mesh_utils.join_scene([oHand, oObj])
                )
                image_utils.save_gif(image_list, save_pref + f"_iter{t:04d}")

        oTh = geom_utils.rt_to_homo(
            geom_utils.rotation_6d_to_matrix(oTh_so6d),
            oTh_trsl,
            oTh_scale,
        )
        hTo = geom_utils.inverse_rt(mat=oTh, return_mat=True)
        nTu_new = nTh @ hTo @ oTu

        return nTu_new.detach(), hA.detach()

    def read_one_grasp(self, sdf_file):
        uSdf, uTo = load_sdf_grid(sdf_file, tensor=True, batched=True, device=device)
        text = open(sdf_file.replace("uSdf.npz", "obj.txt"), "r").read()
        oObj_orig = mesh_utils.load_mesh(
            sdf_file.replace("uSdf.npz", "oObj.obj"), device=device
        )
        nTh = geom_utils.scale_matrix(torch.zeros([1, 3], device=device) + 5)
        oTu = geom_utils.inverse_rt(mat=uTo, return_mat=True)
        nTu = nTh @ oTu
        return [text], uSdf, nTu, oTu, oObj_orig

    def save_grasp(self, nTu, hA, oTu, oObj, loss, save_pref):
        hHand, _ = self.hand_wrapper(None, hA)
        nTh = hand_utils.get_nTh(hA=hA, hand_wrapper=self.hand_wrapper)
        hTn = geom_utils.inverse_rt(mat=nTh, return_mat=True)
        uTo = geom_utils.inverse_rt(mat=oTu, return_mat=True)
        hTo = hTn @ nTu @ uTo
        oTh = geom_utils.inverse_rt(mat=hTo, return_mat=True)
        hObj = mesh_utils.apply_transform(oObj, hTo)
        oHand = mesh_utils.apply_transform(hHand, oTh)

        mesh_utils.dump_meshes([save_pref + "_hObj"], hObj)
        mesh_utils.dump_meshes([save_pref + "_hHand"], hHand)
        mesh_utils.dump_meshes([save_pref + "_oObj"], oObj)
        mesh_utils.dump_meshes([save_pref + "_oHand"], oHand)
        with open(save_pref + "_para.pkl", "wb") as fp:
            pickle.dump(
                {
                    "nTu": nTu.detach().cpu().numpy()[0],
                    "hA": hA.detach().cpu().numpy()[0],
                    "loss": loss,
                },
                fp,
            )


def list_all_inputs(args):
    if args.index is None:
        query = "*"
    else:
        query = args.index
    sdf_list = sorted(glob(osp.join(args.grasp_dir, query, "uSdf.npz")))
    return sdf_list


@main(config_path="configs", config_name="grasp_syn")
@slurm_utils.slurm_engine()
def batch_uniguide(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    uni_guide = UniGuide()
    base_dir = args.save_dir  #  "/home/yufeiy2/scratch/result/uni_guide/"

    sd = uni_guide.init_sds(args, device)
    sdf_list = list_all_inputs(args)

    for t, sdf_file in enumerate(tqdm(sdf_list)):
        if args.get("sds_grasp", False):
            index = sdf_file.split("/")[-2]
            text, uSdf, nTu_fake, oTu, oObj_orig = uni_guide.read_one_grasp(sdf_file)
            cell_list = []

            web_file = osp.join(base_dir, f"{index}_grasp.html")
            for s in range(args.S):
                bs = 1
                save_pref = osp.join(base_dir,  f"{index}_s{s:02d}")

                _, _, nTu_scale = geom_utils.homo_to_rt(nTu_fake)
                rot = geom_utils.random_rotations(bs, device=device)
                r = args.rand_tsl * 5  # in meter --> scale in normalized frame
                tsl = torch.rand([bs, 3], device=device) * r * 2 - r
                npTu = geom_utils.rt_to_homo(rot, tsl, s=nTu_scale)
                npSdf, _ = mesh_utils.transform_sdf_grid(uSdf, npTu, lim=1.5)

                nTnp, hA_pred, pred_loss = uni_guide.sds_grasp(
                    sd,
                    npSdf,
                    text,
                    cfg=args,
                    T=args.T,
                    save_pref=save_pref,
                )
                nTu = nTnp @ npTu

                uni_guide.save_grasp(nTu, hA_pred, oTu, oObj_orig, pred_loss, save_pref)
                line = [
                    pred_loss,
                    save_pref + f"_t{args.T:04d}_pred.gif",
                    save_pref + f"_t{0:04d}_gt.gif",
                ]
                cell_list.append(line)

            # sort row of cell_list by their 1st column
            metric_list = [e[0] for e in cell_list]
            # save metric list to json
            save_file = osp.join(osp.dirname(save_pref), f"{index}_metric.json")
            with open(save_file, "w") as f:
                json.dump(metric_list, f)
            idx = np.argsort(metric_list)
            sorted_cell_list = []
            for i in idx:
                sorted_cell_list.append(cell_list[i])

            sorted_cell_list.insert(
                0,
                [
                    "Final Loss",
                    "Final Grasp",
                    "Init Grasp",
                ],
            )
            web_utils.run(web_file, sorted_cell_list, width=256, inplace=True)

    for t, sdf_file in enumerate(tqdm(sdf_list)):
        if args.get("refine_grasp", False):
            index = sdf_file.split("/")[-2]
            text, uSdf, nTu_fake, oTu, oObj_orig = uni_guide.read_one_grasp(sdf_file)

            index = sdf_file.split("/")[-2]
            grasp_list = sorted(
                glob(osp.join(base_dir, f"{index}_*_para.pkl"))
            )
            for s, grasp_file in enumerate(grasp_list):
                basename = osp.basename(grasp_file).split("_para.pkl")[0]
                w_miss = args.loss.w_miss
                w_pen = args.loss.w_pen
                save_pref = osp.join(base_dir, "refine", basename)
                data = pickle.load(open(grasp_file, "rb"))
                nTu = torch.FloatTensor(data["nTu"]).to(device)[None]
                hA = torch.FloatTensor(data["hA"]).to(device)[None]

                nTu, hA = uni_guide.refine_grasp(
                    oObj_orig,
                    hA,
                    nTu,
                    oTu,
                    save_pref=save_pref,
                    w_pen=w_pen,
                    w_miss=w_miss,
                )

                uni_guide.save_grasp(nTu, hA, oTu, oObj_orig, data["loss"], save_pref)
    return


if __name__ == "__main__":
    batch_uniguide()
