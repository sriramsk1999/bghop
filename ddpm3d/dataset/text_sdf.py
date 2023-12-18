# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------

from random import randint, random
import torch
from torch.utils.data import Dataset
from jutils import geom_utils

class TextSdfDataset(Dataset):
    def __init__(
        self,
        parsed_data,
        side_x=64,
        shuffle=False,
        tokenizer=None,
        text_ctx_len=128,
        uncond_p=0.0,
        uncond_hand=0.,
        use_captions=False,
        is_train=False,
        cfg={},
        data_cfg={},
    ):
        """
        :param parsed_data: a Dict of 'image': [], 'text': [], 'meta': [], 'img_func': 
        :param side_x: _description_, defaults to 64
        :param side_y: _description_, defaults to 64
        :param resize_ratio: _description_, defaults to 0.75
        :param shuffle: _description_, defaults to False
        :param tokenizer: _description_, defaults to None
        :param text_ctx_len: _description_, defaults to 128
        :param uncond_p: _description_, defaults to 0.0
        :param use_captions: _description_, defaults to False
        :param enable_glide_upsample: _description_, defaults to False
        :param upscale_factor: _description_, defaults to 4
        """
        super().__init__()
        self.image_files = parsed_data['image']  # bad name but it's index list
        self.text_files = parsed_data['text']
        self.parsed_data = parsed_data

        self.text_ctx_len = text_ctx_len
        self.is_train = is_train

        self.shuffle = shuffle
        self.side_x = side_x
        self.tokenizer = tokenizer
        self.uncond_p = uncond_p
        self.uncond_hand = uncond_hand
        self.use_captions = use_captions
        self.cfg = cfg
    
        self.weight_list, self.text2num = self.get_weight()
        # weight_list will balance the class/text label
    
    def get_weight(self):
        # count text_files
        text2num = {}
        for text in self.text_files:
            if text not in text2num:
                text2num[text] = 0
            text2num[text] += 1
        cnt_list = [text2num[text] for text in self.text_files]
        weight_list = [1. / cnt for cnt in cnt_list]
        return weight_list, text2num

    def __len__(self):
        return max(1000, len(self.image_files))
    
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def valid_pred(self, x):
        return x is not None and not torch.all(x == 0)
    def augment_with_pred(self, ind, hA, hTo):
        # lots of places to return GT
        if not self.is_train:
            return hA, hTo
        if 'get_anno_pred_fn' not in self.parsed_data:
            return hA, hTo
        if not self.cfg.get('jitter_pred', False):
            return hA, hTo
        
        hA_pred, hTo_pred = self.parsed_data['get_anno_pred_fn'](ind, self.parsed_data['meta'], )
        if self.valid_pred(hA_pred):
            r = torch.rand([1, ])
            hA = geom_utils.interpolate_axisang(hA, hA_pred, r)
            r = torch.rand([1, ])
            hTo_rot_gt, hTo_tsl_gt, _ = geom_utils.matrix_to_axis_angle_t(hTo)
            hTo_rot_pred, hTo_tsl_pred, _ = geom_utils.matrix_to_axis_angle_t(hTo_pred)
            hTo_rot = geom_utils.interpolate_axisang(hTo_rot_gt, hTo_rot_pred, r)
            hTo_tsl = hTo_tsl_gt * r + hTo_tsl_pred * (1-r)
            hTo = geom_utils.axis_angle_t_to_matrix(hTo_rot, hTo_tsl)
        else:
            pass
        return hA, hTo
    
    def __getitem__(self, ind):
        ind = ind % len(self.image_files)
        image_file = self.image_files[ind]
        # read text
        uncond_iter = random() < self.uncond_p
        if self.text_files is None or not self.use_captions or (self.is_train and uncond_iter):
            text = '' 
        else:
            text = self.text_files[ind]

        # linear combination of predicted and GT
        hA, hTo = self.parsed_data['get_anno_fn'](ind, self.parsed_data['meta'], )
        hA, hTo = self.augment_with_pred(ind, hA, hTo)

        # augmentation
        if self.is_train:
            rt = self.cfg.get('jitter_hA_t', 0)
            rs = self.cfg.get('jitter_hA_s', 0)
            hA = hA + torch.randn_like(hA) * rt
            hA = hA * (1 + torch.rand_like(hA) * rs * 2 - rs)

            jit_r = self.cfg.get('jitter_r', 0)
            jit_t = self.cfg.get('jitter_t', 0)
            hpTh = geom_utils.delta_mat(jit_r, jit_t)
            hTo = hpTh @ hTo

        uncond_iter = random() < self.uncond_hand
        if uncond_iter and self.is_train:
            hA = torch.zeros_like(hA)

        try:
            r = self.cfg.get('jitter_x', 0) if self.is_train else 0
            offset = torch.rand([3]) * 2 * r - r
            sdf_dict, nTo = self.parsed_data['img_func'](image_file, ind, self.parsed_data['meta'], offset=offset, hTo=hTo, hA=hA)
        except (FileNotFoundError) as e:
            print(f"An exception occurred trying to load file {image_file}.")
            print(e)
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        out = {
            'text': text,
            'hA': hA[0],
            'offset': offset,
            'nTo': nTo
        }
        out.update(sdf_dict)
        return out