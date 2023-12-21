# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import importlib
import warnings
from typing import Iterable, List

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset

from utils.train_util import get_rank, get_world_size

from .text_sdf import TextSdfDataset


class BalancedDataset(Dataset):
    def __init__(self, dataset, weight_list=None):
        self.dataset = dataset
        self.weight_list = weight_list
        if self.weight_list is None:
            self.weight_list = [1.0 for _ in range(len(dataset))]

        prob_list = torch.FloatTensor(self.weight_list)
        self.unnorm_prob_list = prob_list
        self.max_size = len(self.unnorm_prob_list)

    def __len__(self):
        return self.max_size

    def __getitem__(self, idx):
        ind = torch.multinomial(
            self.unnorm_prob_list,
            1,
        )[0].item()
        return self.dataset[ind]


class ConcatDatasetProb(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset], prob_list=None) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(
                d, IterableDataset
            ), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.all_sizes = [len(e) for e in self.datasets]
        if prob_list is None:
            prob_list = [1.0 for _ in self.datasets]
        print("mix up dataset with prob", prob_list)
        prob_list = torch.FloatTensor(prob_list[0 : len(datasets)])
        self.unnorm_prob_list = prob_list
        self.max_size = max(self.all_sizes)

    def __len__(self):
        return self.max_size
        # return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if len(self.unnorm_prob_list) > 0:
            dataset_idx = torch.multinomial(
                self.unnorm_prob_list,
                1,
            )[0].item()
        else:
            dataset_idx = 0
        sample_idx = idx % self.all_sizes[dataset_idx]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn(
            "cummulative_sizes attribute is renamed to " "cumulative_sizes",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cumulative_sizes


def get_data_parsed(data_cfg, cfg):
    """
    :param data_cfg: a dict with keys: target, data_dir, split, .... defined in data/xxx.yaml
    """
    mod = importlib.import_module("." + data_cfg.target, "ddpm3d.dataset")
    met = getattr(mod, "parse_data")
    if "mix_list" in data_cfg:
        parsed_data = []
        for ds_cfg in data_cfg.mix_list:
            for k in ds_cfg:
                ds_cfg = ds_cfg[k]
                break
            pd = met(ds_cfg.data_dir, ds_cfg.anno_file, ds_cfg, cfg)
            parsed_data.append(pd)

    else:
        parsed_data = met(data_cfg.data_dir, data_cfg.split, data_cfg, cfg)
        parsed_data = [parsed_data]
    return parsed_data


def build_dataloader(
    args, datasets, tokenizer, text_ctx_len, is_train, bs, shuffle=None, workers=10
):
    dataset_list = []
    for ds in datasets:
        data_cfg = datasets[ds]
        data_parserd_list = get_data_parsed(data_cfg, args)
        # print(ds, args.side_x)

        for data_parserd in data_parserd_list:
            dataset = TextSdfDataset(
                data_parserd,
                side_x=args.side_x,
                uncond_p=args.uncond_p,
                uncond_hand=args.get("uncond_hand", True),
                shuffle=shuffle,
                tokenizer=tokenizer,
                text_ctx_len=text_ctx_len,
                use_captions=args.use_captions,
                is_train=is_train,
                cfg=args,
                data_cfg=data_cfg,
            )
            print(dataset_list)
            if args.get("balance_class", False):
                dataset = BalancedDataset(dataset, dataset.weight_list)
            dataset_list.append(dataset)
    if is_train:
        dataset = (
            ConcatDatasetProb(dataset_list, args.get("train_prob", None))
            if len(dataset_list) > 1
            else dataset_list[0]
        )
    else:
        dataset = (
            ConcatDataset(dataset_list) if len(dataset_list) > 1 else dataset_list[0]
        )

    num_tasks = get_world_size()
    global_rank = get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
    )
    print("Sampler_train = %s" % str(sampler_train))

    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        sampler=sampler_train,
        # shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )

    return dataloader
