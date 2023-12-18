
import torch
import torch.nn as nn

from transformers import CLIPTextModel, CLIPTokenizer


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class CLIPTextTokenizer(AbstractEncoder):
    def __init__(self, device="cuda"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="tokenizer",
            cache_dir='/home/yufeiy2/scratch/pretrain')
        self.device = device

    @torch.no_grad()    
    def forward(self, text):
        batch_encoding = self.tokenizer(text, 
            truncation=True, 
            max_length=self.tokenizer.model_max_length, 
            padding="max_length", return_tensors="pt")
        batch_encoding = {k: v.to(self.device) for k, v in batch_encoding.items()}
        return batch_encoding


class CLIPPooledEmbedder(nn.Module):
    def __init__(self, device='cuda', **kwargs) -> None:
        super().__init__()
        self.device = device
        self.tknz_fn = CLIPTextTokenizer(device)
        self.model = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder", 
            cache_dir='/home/yufeiy2/yufeiy2/pretrain/')
        self.ndim = self.model.config.hidden_size

    def forward(self, text, use_tokenizer=True):
        if use_tokenizer:
            tokens = self.tknz_fn(text)
        else:
            tokens = text
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        z = self.model(**tokens)
        z = z.pooler_output.unsqueeze(1)
        # (N, 1, 768)
        return z

    def encode(self, text):
        return self(text)