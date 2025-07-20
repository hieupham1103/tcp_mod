import os.path as osp
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict
import scipy.io as sio


from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from .clip_text import clip
from .clip_text.simple_tokenizer import SimpleTokenizer as _Tokenizer
import tqdm
_tokenizer = _Tokenizer()
import numpy as np
import copy




def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # Add design details for MMA
    design_details = {"trainer": "TCP_MOD_MMA"}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

CUSTOM_TEMPLATES_ori = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of an aircraft {}.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of a {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

CUSTOM_TEMPLATES = {
    "OxfordPets": "X X X X {}, a type of pet.",
    "OxfordFlowers": "X X X X {}, a type of flower.",
    "FGVCAircraft": "X X X X {}, a type of aircraft.",
    "DescribableTextures": "X X X X {} texture.",
    "EuroSAT": "X X X X {}.",
    "StanfordCars": "X X X X {}, a type of car",
    "Food101": "X X X X {}, a type of food.",
    "SUN397": "X X X X {}.",
    "Caltech101": "X X X X {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


class ImageEncoder(nn.Module):
    def __init__(self, clip_visual):
        super().__init__()
        for attr in dir(clip_visual):
            if not attr.startswith('_') and not callable(getattr(clip_visual, attr)):
                setattr(self, attr, getattr(clip_visual, attr))
        
        for name, module in clip_visual.named_modules():
            if name:
                setattr(self, name, module)
    
    def forward(self, x: torch.Tensor, visual_adapter_func=None):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        # Check if transformer supports adapter function
        if visual_adapter_func is not None and hasattr(self.transformer, 'forward') and len(inspect.signature(self.transformer.forward).parameters) > 1:
            x = self.transformer(x, visual_adapter_func)
        else:
            x = self.transformer(x)
        x = x.permute(1, 0, 2)

        cls_token = x[:, 0, :]
        patch_tokens = x[:, 1:, :]
        
        cls_token = self.ln_post(cls_token)
        
        if self.proj is not None:
            cls_token = cls_token @ self.proj
            
        return cls_token, patch_tokens


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, class_feature, weight, tokenized_prompts, text_adapter_func=None, flag=False):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if flag:
            # Standard transformer with adapter support
            if text_adapter_func is not None and hasattr(self.transformer, 'forward') and len(inspect.signature(self.transformer.forward).parameters) > 1:
                x = self.transformer(x, text_adapter_func)
            else:
                x = self.transformer(x)
        else:
            # TCP-style processing - for now, we'll use the TCP approach
            # TODO: In future, we could integrate adapters into the TCP resblocks
            counter=0
            outputs = self.transformer.resblocks([x,class_feature,weight,counter])
            x = outputs[0]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            print("use given words to initialize context vectors")
            temp = 'a photo of a'
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

            ctx_vectors_src = embedding[0, 1 : 1 + n_ctx, :]

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)


        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        clip_model_ = load_clip_to_cpu(cfg)
        clip_model_.cuda()

        temp = CUSTOM_TEMPLATES_ori[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        vis_dim = clip_model.visual.output_dim
        self.meta_net = nn.Sequential(
            OrderedDict([("linear1", nn.Linear(vis_dim, vis_dim // 4,bias=True)),
                         ("relu", QuickGELU()),
                         ("linear2", nn.Linear(vis_dim // 4, 4*ctx_dim,bias=True))
                         ]))
        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()
            
        classnames = [name.replace("_", " ") for name in classnames]
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(prompts)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.prev_ctx=None

    def forward(self, text_features):
        # import pdb; pdb.set_trace()
        text_features = text_features.mean(dim=0)
        class_feature = self.meta_net(self.text_features)
        class_feature = class_feature.reshape(class_feature.shape[0],-1,512)
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompt = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompt, class_feature

class Attention(nn.Module):
    def __init__(self, image_c, text_c, n_heads=8):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(text_c, n_heads, batch_first=True)
        self.image_proj = nn.Linear(image_c, text_c)

    def forward(self, image_features, text_features):
        image_features = self.image_proj(image_features)
        text_features = text_features.unsqueeze(0).repeat(image_features.shape[0], 1, 1)
        x = self.attention(text_features, image_features, image_features)[0]
        return x
    
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

from scipy.optimize import linear_sum_assignment

class AdapterLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.n_cls = len(classnames)
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # build multi-modal adapter
        self.text_adapter_func = lambda x: self.return_text_adapter(index=x)
        self.text_adapter = self._build_adapter(
            clip_model.ln_final.weight.shape[0], 
            len(clip_model.transformer.resblocks), 
            cfg.TRAINER.TCP_MOD_MMA.ADAPTER_START,
            cfg.TRAINER.TCP_MOD_MMA.ADAPTER_END,
            cfg.TRAINER.TCP_MOD_MMA.ADAPTER_DIM,
            clip_model.dtype
        )
        
        self.visual_adapter_func = lambda x: self.return_visual_adapter(index=x)
        self.visual_adapter = self._build_adapter(
            clip_model.visual.ln_post.weight.shape[0],
            len(clip_model.visual.transformer.resblocks), 
            cfg.TRAINER.TCP_MOD_MMA.ADAPTER_START,
            cfg.TRAINER.TCP_MOD_MMA.ADAPTER_END,
            cfg.TRAINER.TCP_MOD_MMA.ADAPTER_DIM,
            clip_model.dtype
        )

        self.shared_adapter = self._build_adapter(
            cfg.TRAINER.TCP_MOD_MMA.ADAPTER_DIM,
            len(clip_model.visual.transformer.resblocks), 
            cfg.TRAINER.TCP_MOD_MMA.ADAPTER_START,
            cfg.TRAINER.TCP_MOD_MMA.ADAPTER_END,
            cfg.TRAINER.TCP_MOD_MMA.ADAPTER_DIM,
            clip_model.dtype
        )
        self.adapter_scale = float(cfg.TRAINER.TCP_MOD_MMA.ADAPTER_SCALE)

    def return_text_adapter(self, index):
        if index < len(self.text_adapter) and self.text_adapter[index] is not None:
            return self.text_adapter[index], self.shared_adapter[index], self.adapter_scale
        return None, None, self.adapter_scale

    def return_visual_adapter(self, index):
        if index < len(self.visual_adapter) and self.visual_adapter[index] is not None:
            return self.visual_adapter[index], self.shared_adapter[index], self.adapter_scale
        return None, None, self.adapter_scale

    def _build_adapter(self, d_model, n_layers, l_start, l_end, mid_dim, dtype):
        adapter = [None] * (n_layers + 1)
        for i in range(l_start, l_end+1):
            if mid_dim == d_model:
                adapter[i] = nn.Sequential(
                    nn.Linear(d_model, mid_dim),
                    nn.ReLU()
                )
            else:
                adapter[i] = nn.Sequential(OrderedDict([
                    ("down", nn.Sequential(nn.Linear(d_model, mid_dim), nn.ReLU())),
                    ("up", nn.Linear(mid_dim, d_model))
                ]))
        adapter = nn.ModuleList([a for a in adapter])
        for m in adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

        if dtype == torch.float16:
            for m in adapter.modules():
                if m is not None:
                    m.half()
    
        return adapter
    
    def forward(self):
        # Simply return the adapter functions - no text embedding processing needed
        return None, self.text_adapter_func, self.visual_adapter_func

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.adapter_learner = AdapterLearner(cfg, classnames, clip_model)
        self.cross_attn_text_img = Attention(768, 512, n_heads=8)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = self.prompt_learner.text_features
        self.image_encoder = ImageEncoder(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.domain_sim = -1
        self.domain_sim_src = -1
        self.weight = cfg.TRAINER.COOP.W
        
        if self.dtype == torch.float16:
            self.cross_attn_text_img.half()
    
    def forward(self, image, label=None):
        # Get adapter functions
        _, text_adapter_func, visual_adapter_func = self.adapter_learner()
        
        # Use visual adapter in image encoding
        cls_token, patch_tokens = self.image_encoder(image.type(self.dtype), visual_adapter_func)
        text_features_old = self.ori_embedding
        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
        patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)
        
        atten_features = self.cross_attn_text_img(patch_tokens, text_features_old)
        prompts,class_prompt = self.prompt_learner(atten_features)
        
        # Use text adapter in text encoding (use standard transformer mode for adapter support)
        text_features = self.text_encoder(prompts, class_prompt, self.weight, tokenized_prompts.detach(), text_adapter_func, flag=True) 
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale.detach() * cls_token.detach() @ text_features_norm.t()
        
        
        if self.prompt_learner.training:
            score= cos(text_features_norm,text_features_old)
            score  = 1.0-torch.mean(score)
            loss = F.cross_entropy(logits, label)+8.0*score
            return logits, loss
        else:
            return logits


@TRAINER_REGISTRY.register()
class TCP_MOD_MMA(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(classnames)
        self.n_cls = len(classnames)
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.w = cfg.TRAINER.COOP.W
        
        print("Turning off gradients in both the image and the text encoder")
        
        for _, param in self.model.named_parameters():
            param.requires_grad_(False)

        name_to_update = ["prompt_learner", "cross_attn_text_img", "adapter_learner"]
        for name, param in self.model.named_parameters():
            for n2u in name_to_update:
                if n2u in name:
                    param.requires_grad_(True)
                    print(f"Training parameter: {name}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: include adapter_learner in the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("cross_attn_text_img", self.model.cross_attn_text_img, self.optim, self.sched)
        self.register_model("adapter_learner", self.model.adapter_learner, self.optim, self.sched)
        
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        self.proto=None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output,loss = self.model(image, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    #def model_inference(self, input):
    #    return self.model(input)


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print(names)

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
