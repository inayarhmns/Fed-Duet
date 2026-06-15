import os.path as osp
import os
import time
import random

import numpy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import copy
from collections import defaultdict

# === Prompt Pool & Gate imports ===
from .prompt_pool import PromptPool, GateNetwork, SparseGateLoss

from continual_clip.sampling import sample_iid, sample_noniid
import clip.clip as clip
from clip import model as clip_module
from continual_clip.utils import get_class_ids_per_task, get_class_names
from clip.tokenizer import SimpleTokenizer as _Tokenizer

from . import utils

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    """Encode prompts with the CLIP text transformer."""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        transformer_output = self.transformer(x)
        loss = None
        if isinstance(transformer_output, tuple):
            x = transformer_output[0]
            loss = transformer_output[1]
        else:
            x = transformer_output
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        if loss is not None:
            return x, loss
        else:
            return x


class PromptLearner(nn.Module):
    """Client-side prompt learner for local prompt vectors. This is CoOp-style prompt vector maker"""

    def __init__(self, cfg, classnames, clip_model, prev_ctx=None):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_cls = len(classnames)
        n_ctx = getattr(cfg, "N_CTX", 16)
        ctx_init = getattr(cfg, "CTX_INIT", "")
        csc = getattr(cfg, "CSC", False)
        class_token_position = getattr(cfg, "CLASS_TOKEN_POSITION", "end")

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if prev_ctx is not None:
            ctx_vectors = prev_ctx.to(torch.float32)
            prompt_prefix = " ".join(["X"] * n_ctx)
        elif ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :].to(device)
            ctx_vectors = ctx_vectors.to(torch.float32)
            prompt_prefix = ctx_init
        else:
            if csc:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=torch.float32, device=device)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=torch.float32, device=device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        print(f"[PromptLearner] Prompts: {prompts}")

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(device)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = class_token_position

    @property
    def ctx_base(self):
        return self.ctx

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        ctx = ctx.to(prefix.dtype)

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i],
                                   dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts


# --------------------------------------------------
# CustomCLIP
# --------------------------------------------------

class CustomCLIP(nn.Module):
    """Custom CLIP wrapper that embeds prompt learning and text encoding."""

    def __init__(self, cfg, classnames, clip_model, prev_ctx=None, prev_fusion_state=None):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, prev_ctx)
        self.cfg = cfg
        self.n_class = len(classnames)
        self.classnames = classnames

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.clip_model = clip_model
        self.visual = clip_model.visual
        self.transformer = clip_model.transformer

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.ln_final = clip_model.ln_final
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.text_projection = clip_model.text_projection

        self.expert_prompts: torch.Tensor = None  # (K_expert, 512)

        feature_dim = clip_model.ln_final.weight.shape[0]
        gating_embed_dim = getattr(cfg, "gating_embed_dim", 128)
        num_heads = getattr(cfg, "gating_heads", 8)
        scaling = getattr(cfg, "gating_scaling", 10.0)

        # Ensure feature_dim is divisible by gating_embed_dim for integer reduce_times
        if feature_dim % gating_embed_dim != 0:
            raise ValueError(f"Feature dim ({feature_dim}) must be divisible by gating_embed_dim ({gating_embed_dim})")
        self.reduce_times = feature_dim // gating_embed_dim

        self.fusion_gating = MultiheadAttention(gating_embed_dim, num_heads=num_heads, scaling=scaling, dtype=torch.float32)

        if prev_fusion_state is not None:
            try:
                self.fusion_gating.load_state_dict(prev_fusion_state, strict=False)
            except Exception as e:
                print(f"[CustomCLIP] is using randomly initialized fusion_gating.")

        self.lmbda = getattr(cfg, "ctx_lmbda", 0.5)

        self.nonlocal_ctx = None
        self.nonlocal_text_features = []

        self.experts_num = cfg.num_experts


    def pool(self, t: torch.Tensor):
        """Feature pooling by slicing."""
        if len(t.shape) == 4:
            return t[:, :, :, ::self.reduce_times]
        if len(t.shape) == 3:
            return t[:, :, ::self.reduce_times]
        if len(t.shape) == 2:
            return t[:, ::self.reduce_times]
        return None

    def load_ctx(self, ctx: torch.Tensor):
        """Load the given ctx (n_ctx, d) into prompt_learner for encoding."""
        state_dict = self.prompt_learner.state_dict()
        state_dict['ctx'] = ctx
        self.prompt_learner.load_state_dict(state_dict, strict=False)

    def _compute_nonlocal_text_features(self):
        if self.nonlocal_ctx is None:
            self.nonlocal_text_features = []
            return

        temp_local_state = copy.deepcopy(self.prompt_learner.state_dict())
        self.nonlocal_text_features = []

        if not isinstance(self.nonlocal_ctx, list):
            self.nonlocal_ctx = [self.nonlocal_ctx]

        for ctx in self.nonlocal_ctx:
            # load nonlocal ctx
            self.load_ctx(ctx)

            # compute nonlocal text features
            with torch.no_grad():
                text_output = self.text_encoder(self.prompt_learner(), self.tokenized_prompts)
                text_features = text_output if isinstance(text_output, torch.Tensor) else text_output[0]

                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = self.pool(text_features)
                self.nonlocal_text_features.append(text_features.detach())

        self.prompt_learner.load_state_dict(temp_local_state)

    def update_prompt_learner(self, prev_ctx=None, new_classnames=None): # trigger making the promptlearner
        need_reinit = (new_classnames is not None) or (prev_ctx is not None)

        if not need_reinit:
            return

        if prev_ctx is None:
            prev_ctx = self.prompt_learner.ctx.detach().clone()

        if new_classnames is None:
            new_classnames = self.classnames


        self.prompt_learner = PromptLearner(self.cfg, new_classnames, self.clip_model, prev_ctx)

        self.classnames = new_classnames
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        print(f"tokenized prompts: {self.tokenized_prompts}")

    def forward(self, image, text=None, task_id=None, is_train=True, prev_ctx=None):
        if task_id is not None:
            clip_module.global_taskid = task_id
            if hasattr(clip_module, "global_gate_collector"):
                clip_module.global_gate_collector['current_task_id'] = task_id
                clip_module.global_gate_collector['image'] = {}
                clip_module.global_gate_collector['text'] = {}

        with torch.no_grad():
            ctx_eff_vec = self.prompt_learner.ctx.detach().clone()  # [n_ctx, d_ctx]

        if hasattr(self.image_encoder, "set_ctx_vector"):
            self.image_encoder.set_ctx_vector(ctx_eff_vec)

        image_output = self.image_encoder(image.type(self.dtype))
        if isinstance(image_output, tuple):
            image_features, image_loss = image_output
        else:
            image_features = image_output
            image_loss = None

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # (B, D)

        local_prompts = self.prompt_learner()
        tokenized = self.tokenized_prompts
        
        text_batch_size = getattr(self.cfg, "text_batch_size", 256)
        num_prompts = local_prompts.shape[0]
        
        all_text_features = []
        all_text_losses = []
        
        for i in range(0, num_prompts, text_batch_size):
            prompt_chunk = local_prompts[i : i + text_batch_size]
            token_chunk = tokenized[i : i + text_batch_size]
            
            text_output_chunk = self.text_encoder(prompt_chunk, token_chunk)
            
            if isinstance(text_output_chunk, tuple):
                text_features_chunk, text_loss_chunk = text_output_chunk
                if text_loss_chunk is not None:
                    all_text_losses.append(text_loss_chunk)
            else:
                text_features_chunk = text_output_chunk
            
            all_text_features.append(text_features_chunk)

        text_features = torch.cat(all_text_features, dim=0)
        text_loss = torch.stack(all_text_losses).mean() if all_text_losses else None
        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        local_logits = logit_scale * image_features @ text_features.t()  # (B, n_cls)

        if self.training and self.nonlocal_text_features:
            q = self.pool(image_features).repeat(self.n_class, 1, 1)

            global_expert_feats = torch.stack(self.nonlocal_text_features, dim=0)  # (n_experts, n_cls, d_pooled)
            k = v = torch.cat([
                self.pool(text_features).unsqueeze(1),  # (n_cls, 1, d_pooled)
                global_expert_feats.permute(1, 0, 2)  # (n_cls, n_experts, d_pooled)
            ], dim=1)

            orig_dtype = q.dtype
            new_features, _ = self.fusion_gating(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32))  # (n_cls, B, d_pooled)
            new_features = new_features.to(orig_dtype)
            
            new_features = new_features.permute(1, 2, 0)  # (B, d_pooled, n_cls)

            fused_logits = logit_scale * torch.bmm(self.pool(image_features).unsqueeze(1), new_features).squeeze(1)

            logits = self.lmbda * local_logits + (1 - self.lmbda) * fused_logits
        else:
            logits = local_logits

        total_loss = None
        if image_loss is not None or text_loss is not None:
            total_loss = 0.0
            if image_loss is not None:
                total_loss += image_loss
            if text_loss is not None:
                total_loss += text_loss

        return logits, total_loss

    @torch.no_grad()
    def set_global_experts(self, expert_ctx_list):
        """服务器下发全局专家 Prompt ctx 列表 (List[Tensor] 或 Tensor)。"""
        self.nonlocal_ctx = expert_ctx_list
        self._compute_nonlocal_text_features()


# --------------------------------------------------
# FedDuetTrainer
# --------------------------------------------------

class FedDuetTrainer:

    def __init__(self, cfg, global_model, client_model, train_dataset, eval_dataset, task_id, texts, prev_ctx=None,
                 prev_fusion_state=None, prev_mean_acc_history=None, classes_names=None, prev_client_states=None):
        self.cfg = cfg
        self.global_model = CustomCLIP(cfg, texts, global_model, prev_ctx, prev_fusion_state)
        self.client_model = CustomCLIP(cfg, texts, client_model, prev_ctx, prev_fusion_state)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.task_id = task_id
        self.texts = texts
        self.classes_names = classes_names
        if hasattr(cfg, "device"):
            self.device = cfg.device if isinstance(cfg.device, torch.device) else torch.device(cfg.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.global_model.to(torch.device("cpu"))
        self.client_model.to(self.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.iid = getattr(cfg, "iid", True)
        self.num_clients = getattr(cfg, "num_clients", 5)
        self.com_rounds = getattr(cfg, "com", 10)
        self.client_epochs = getattr(cfg, "client_epochs", 5)
        self.current_round = 0
        self.metrics = defaultdict(list)


        self.mean_acc_history = prev_mean_acc_history.copy() if prev_mean_acc_history else []

        self.final_client_states = [{} for _ in range(self.num_clients)]

        if self.iid:
            self.dict_users = sample_iid(train_dataset[task_id:task_id + 1], self.num_clients)
        else:
            self.dict_users = sample_noniid(train_dataset[task_id:task_id + 1], self.num_clients)

        self.clients_loaders = []
        self.client_sizes = []
        for uid in range(self.num_clients):
            client_indices = list(self.dict_users[uid])
            client_subset = Subset(train_dataset[task_id:task_id + 1], client_indices)
            client_loader = DataLoader(
                client_subset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=getattr(cfg, "num_workers", 1),
                drop_last=True
            )
            self.clients_loaders.append(client_loader)
            self.client_sizes.append(len(client_subset))

            # Print shapes and tensor data
            print("Features Batch Shape client_loader:", client_loader.dataset[0][0].shape) # for example, 3, 224, 224. Means 3 channels, 224 height, 224 width.
            print("Labels Batch Shape client_loader:", client_loader.dataset[0][1].shape)
            print("Actual Data Samples client_loader:\n", client_loader.dataset[0][0])

        self.prompt_pool_size = getattr(cfg, "prompt_pool_size", 64)
        with torch.no_grad():
            base_ctx = self.global_model.prompt_learner.ctx.detach().clone()  # (n_ctx,d)
        
        if getattr(cfg, "scenario", "class") == "domain":
            class_file_path = "cil/dataset_reqs/domainnet_classes.txt"
        else:
            class_file_path = "cil/dataset_reqs/imagenet1000_classes.txt"
        # here the server makes the prompt pool by kmeans of the classnames if CIL or domain names if DIL.
        self.prompt_pool = PromptPool(
            K=self.prompt_pool_size,
            base_prompt=base_ctx,
            init_by_kmeans=getattr(cfg, "init_pool_by_kmeans", True),
            clip_model=self.global_model.clip_model,
            class_file=class_file_path
        )
        print(f"self.prompt_pool_size: {self.prompt_pool_size}, base_ctx shape: {base_ctx.shape}") # for example, base_ctx shape: torch.Size([16, 512]). Means n_ctx=16, ctx_dim=512.
        self.num_experts_per_client = getattr(cfg, "num_experts", 8) # this is for fine-grained experts per client

        self.feature_dim = base_ctx.shape[-1]
        gate_hidden = getattr(cfg, "gate_hidden_dim", 512)
        self.gate = GateNetwork(self.feature_dim, self.prompt_pool_size, gate_hidden).to(self.device)
        self.gate_optimizer = torch.optim.Adam(self.gate.parameters(), lr=getattr(cfg, "gate_lr", 1e-3))

        self._gate_train_buffer = []

        self.client_features = [None] * self.num_clients

        moe_keywords = ("adaptmlp_list", "router", "noise", "shared_expert")
        # if the config explicitly says not to unfreeze moe params, then freeze them. By default, we unfreeze them for training
        unfreeze_moe = getattr(cfg, "unfreeze_moe", True)
        if not unfreeze_moe:
            for model in (self.global_model, self.client_model):
                for n, p in model.named_parameters():
                    if any(k in n for k in moe_keywords):
                        p.requires_grad = False

        self.upload_moe_params = getattr(cfg, "upload_moe_params", True)
        # if unfreeze_moe is set True in config, unfreeze moe_keywords adapters 
        if (not self.upload_moe_params) and (not unfreeze_moe):
            for model in (self.global_model, self.client_model):
                for n, p in model.named_parameters():
                    if any(k in n for k in moe_keywords):
                        p.requires_grad = False

        self.gradient_clip_norm = getattr(cfg, "gradient_clip_norm", 1.0)

        self.dict_users_test_all_tasks = []
        if self.eval_dataset is not None:
            for t in range(len(self.eval_dataset)):
                task_dataset = self.eval_dataset[t:t+1]
                if self.iid:
                    self.dict_users_test_all_tasks.append(sample_iid(task_dataset, self.num_clients))
                else:
                    self.dict_users_test_all_tasks.append(sample_noniid(task_dataset, self.num_clients))

        self._moe_param_names = [n for n, _ in self.global_model.named_parameters() if
                                 any(k in n for k in moe_keywords)]

    def _train_client(self, client_id, train_loader, prompt_idx):
        """Training a single client model"""
        self.client_model.train()

        prompt_keywords = ("prompt_learner", "fusion_gating")
        moe_keywords = ("adaptmlp_list", "router", "noise", "shared_expert")

        for p in self.client_model.parameters():
            p.requires_grad = False
        # parametric pathway unfreezing
        if self.current_round < self.com_rounds // 2:
            print(f"[Client {client_id}] Round {self.current_round}: Train Parametric Experts")
            for name, p in self.client_model.named_parameters():
                print(f"[PARAMETRIC] Unfreezing {name} for training.")
                if any(k in name for k in moe_keywords):
                    p.requires_grad = True
        # semantic pathway unfreezing
        else:
            print(f"[Client {client_id}] Round {self.current_round}: Train Semantic Experts")
            for name, p in self.client_model.named_parameters():
                if any(k in name for k in prompt_keywords):
                    print(f"[SEMANTIC] Unfreezing {name} for training.")
                    p.requires_grad = True


        # ---- Step 2: compute feature summary f_i over few batches ----
        def _feature_summary(max_batches: int = 10):
            feats = []
            cnt = 0
            for imgs, _, _ in train_loader:
                imgs = imgs.to(self.device)
                with torch.no_grad():
                    out = self.client_model.image_encoder(imgs.type(self.client_model.dtype))
                    if isinstance(out, tuple):
                        out = out[0]
                    out = out / out.norm(dim=-1, keepdim=True)
                    feats.append(out.mean(dim=0))
                cnt += 1
                if cnt >= max_batches:
                    break

            feat_vec = torch.stack(feats).mean(dim=0)  # (d,)

            if getattr(self.cfg, "enable_dp", False):
                clip_norm = getattr(self.cfg, "dp_clip", 1.0)
                noise_mul = getattr(self.cfg, "dp_noise_multiplier", 0.1)

                norm = feat_vec.norm()
                if norm > clip_norm:
                    feat_vec = feat_vec * (clip_norm / norm)

                if noise_mul > 0:
                    noise = torch.randn_like(feat_vec) * (clip_norm * noise_mul)
                    feat_vec = feat_vec + noise

                print(f"[Client {client_id}] DP: clip={clip_norm}, noise_mul={noise_mul}")

            return feat_vec

        feat_summary = _feature_summary(max_batches=getattr(self.cfg, "summary_batches", 1))  # (d,)

        trainable_params = [p for p in self.client_model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.cfg.lr,
            weight_decay=getattr(self.cfg, "weight_decay", 0.01)
        )
        scaler = torch.cuda.amp.GradScaler()
        # accumulation_steps is a trick to simulate a larger batch size when the GPU doesn't have enough memory to fit a large batch. 
        # So the gradients are accumulated over multiple iterations before performing an optimizer step. 
        #For example, if accumulation_steps=4, the model will accumulate gradients for 4 batches before updating the weights.         
        accumulation_steps = getattr(self.cfg, "gradient_accumulation_steps", 1)

        total_iterations = len(train_loader) * self.client_epochs # len(train_loader) is the number of batches, not the batch_size 
        scheduler = utils.cosine_lr(optimizer, self.cfg.lr, 30, total_iterations)
        train_iter = iter(train_loader)
        progress_bar = tqdm(range(total_iterations), desc=f"Client {client_id} is trained on task {self.task_id})")

        client_metrics = defaultdict(list)

        running_accuracy = 0.0
        running_loss = 0.0
        
        optimizer.zero_grad()

        for iteration in range(total_iterations): 
            try:
                inputs, targets, task_ids = next(train_iter) # this processes each batch, with each of the size batch_size.
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets, task_ids = next(train_iter)

            if getattr(self.cfg, "scenario", "class") == "class" and hasattr(self.cfg, "increment"):
                shift = self.task_id * self.cfg.increment
                targets = targets - shift

            inputs = inputs.to(self.device)
            targets = targets.to(self.device).long()


            with torch.cuda.amp.autocast():
                output, moe_loss = self.client_model(inputs, task_id=self.task_id) # forward

                cls_loss = F.cross_entropy(output, targets, label_smoothing=getattr(self.cfg, "ls", 0.0))
                total_loss = cls_loss
                if moe_loss is not None:
                    total_loss += moe_loss
                if accumulation_steps > 1:
                    total_loss = total_loss / accumulation_steps

                with torch.no_grad():
                    _, predicted = torch.max(output, 1)
                    batch_accuracy = (predicted == targets).float().mean().item() # accuracy is averaged within batch
                    running_accuracy = 0.9 * running_accuracy + 0.1 * batch_accuracy # weighted accuracy of current running accuracy and the batch (averaged) accuracy. 
                    running_loss = 0.9 * running_loss + 0.1 * total_loss.item() * accumulation_steps
                    print(f"[Client {client_id}] Iteration {iteration + 1}/{total_iterations} | Loss: {running_loss:.4f}, Acc: {running_accuracy:.4f}")
            scaler.scale(total_loss).backward()

            # every iteration, DO FORWARD AND BACKWARD. but the gradient updates are done after every accumulation_steps iterations, or at the end of training. This simulates a larger batch size and can help stabilize training when GPU memory is limited.
            if (iteration + 1) % accumulation_steps == 0 or (iteration + 1) == total_iterations: # gradient updates are done every accumulation_steps iterations, or at the end of training
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            scheduler(iteration)

            client_metrics["loss"].append(total_loss.item() * accumulation_steps)
            client_metrics["accuracy"].append(batch_accuracy)

            if moe_loss is not None:
                if isinstance(moe_loss, float):
                    moe_loss = torch.tensor(moe_loss, device=self.device)
                client_metrics["moe_loss"].append(moe_loss.item())
            progress_bar.set_postfix({
                'loss': f"{running_loss:.4f}",
                'acc': f"{running_accuracy:.4f}",
                'moe_loss': f"{moe_loss.item() if moe_loss is not None else 0:.4f}"
            })
            progress_bar.update(1)

            del inputs, targets, output, total_loss

        progress_bar.close()
        avg_metrics = {
            "loss": np.mean(client_metrics["loss"]) if client_metrics["loss"] else 0.0,
            "accuracy": np.mean(client_metrics["accuracy"]) if client_metrics["accuracy"] else 0.0
        }
        self._gate_train_buffer.append((feat_summary.detach().cpu(), prompt_idx, avg_metrics["loss"]))

        # print(f"[客户端 {client_id}, 任务 {self.task_id}] 已完成 {total_iterations} 次迭代")
        # print(f"最终指标 - 损失: {avg_metrics['loss']:.6f}, 准确率: {avg_metrics['accuracy']:.6f}")

        for key, values in client_metrics.items():
            if values:
                self.metrics[key].extend(values)

        return avg_metrics, feat_summary.detach()

    def train(self):
        self.global_model.train()

        for global_round in range(self.com_rounds):
            self.current_round = global_round
            print(f"\n=== Global Round [{global_round + 1}/{self.com_rounds}] ===")

            client_states = []
            client_metrics = []

            for client_id, client_loader in enumerate(self.clients_loaders):
                print(f"\n---Client  {client_id + 1}/{len(self.clients_loaders)} Training ---")

                global_state_on_device = {k: v.to(self.device) for k, v in self.global_model.state_dict().items()}
                self.client_model.load_state_dict(global_state_on_device, strict=False)

                indices = random.sample(range(self.prompt_pool_size), self.num_experts_per_client)
                expert_ctx_list = [self.prompt_pool.prompts[idx].to(self.device) for idx in indices]
                # print(f"[Server] Round {global_round} | Client {client_id} 分配专家索引: {indices}")

                if self.client_features[client_id] is not None:
                    with torch.no_grad():
                        logits_pred = self.gate(self.client_features[client_id].to(self.device).to(torch.float32))
                    top_indices = logits_pred.argsort(descending=True)[:self.num_experts_per_client]
                    indices = top_indices.cpu().tolist()
                    print(f"top_indices for client {client_id}: {indices}")
                else:
                    # cold start
                    indices = random.sample(range(self.prompt_pool_size), self.num_experts_per_client)

                expert_ctx_list = [self.prompt_pool.prompts[idx].to(self.device) for idx in indices]
                # print(f"[Server] Round {global_round} | Client {client_id} 分配专家索引: {indices}")
                print(f"indices for client {client_id}: {indices}")
                print(f"expert_ctx_list shape for client {client_id}: { expert_ctx_list.shape if isinstance(expert_ctx_list, torch.Tensor) else [ctx.shape for ctx in expert_ctx_list]}")
                print(f"expert_ctx_list for client {client_id}: {expert_ctx_list}")


                self.client_model.set_global_experts(expert_ctx_list)

                client_metric, feat_summary = self._train_client(client_id, client_loader, indices)
                print(f"[Client {client_id}] client_metric: {client_metric}")
                print(f"[Client {client_id}] feat_summary: {feat_summary}")

                client_metrics.append(client_metric)
        
                self.client_features[client_id] = feat_summary.detach().cpu()

                is_final_round = global_round == self.com_rounds - 1

                # taking moe params here
                moe_state = {
                    n: p.detach().cpu() for n, p in self.client_model.named_parameters() if any(k in n for k in self._moe_param_names)}
                client_state_for_agg = {'moe': moe_state} # moe params are collected every round and will be sent back to the server

                p_params_for_current_agg = {}


                if is_final_round: # if it is the final round, collect the state of the client parameters
                    print(f"[Client {client_id}] In the final round, all personalized parameters are collected for the final aggregation.")
                    final_p_state = {
                        n: p.detach().cpu()
                        for n, p in self.client_model.named_parameters()
                        if any(k in n for k in ("prompt_learner", "fusion_gating", "adaptmlp_list"))
                    }
                    p_params_for_current_agg.update(final_p_state)
                # personalized here are model states that are sent back to server, only averaged and updated in the final round and saved in each client
                if p_params_for_current_agg:
                    client_state_for_agg['personalized'] = p_params_for_current_agg
                # collecting client_state after training
                client_states.append(client_state_for_agg)


                personalized_keywords = ("prompt_learner", "fusion_gating", "adaptmlp_list")
                final_p_state = {
                    n: p.detach().cpu().clone()
                    for n, p in self.client_model.named_parameters()
                    if any(k in n for k in personalized_keywords)
                }
                self.final_client_states[client_id] = final_p_state


            global_state = self.global_model.state_dict()
            total_size = sum(self.client_sizes)
            client_weights = [size / total_size for size in self.client_sizes]

            if self.current_round < self.com_rounds // 2:
                print("\n---  Parametric Experts Parameter ---")
                shared_moe_keywords = ("shared_expert", "router", "noise")

                for key in self._moe_param_names:
                    if any(k in key for k in shared_moe_keywords):
                        if all('moe' in state and key in state['moe'] for state in client_states):
                            weighted_sum = torch.zeros_like(global_state[key])
                            for i, state in enumerate(client_states):
                                weighted_sum += client_weights[i] * state['moe'][key].to(global_state[key].device)
                            global_state[key] = weighted_sum
            # this aggregates the personalized parameters
            if client_states and 'personalized' in client_states[0]:
                print("\n--- aggregate Personalized parameters ---")
                personal_params_to_agg = client_states[0]['personalized'].keys()

                for key in personal_params_to_agg:
                    if all('personalized' in state and key in state['personalized'] for state in client_states):
                        weighted_sum = torch.zeros_like(global_state[key])
                        for i, state in enumerate(client_states):
                            weighted_sum += client_weights[i] * state['personalized'][key].to(global_state[key].device)
                        global_state[key] = weighted_sum

            self.global_model.load_state_dict(global_state)
            
            for metric_name in ["loss", "accuracy"]:
                values = [m[metric_name] for m in client_metrics]

                if isinstance(values[0], (list, tuple)):
                    values = [numpy.mean(v) for v in values]

                weighted_avg = sum(v * w for v, w in zip(values, client_weights))
                print(f"Global Round {global_round + 1} Average {metric_name}: {weighted_avg:.6f}")

            self._update_gate_network()
        self.evaluate_clients()
        # ----------------------------------------------------

        return self.global_model.clip_model, self.global_model.prompt_learner.ctx.detach().clone(), \
               self.global_model.fusion_gating.state_dict(), self.mean_acc_history, self.final_client_states

    def evaluate_clients(self):

        from torch.utils.data import ConcatDataset, Subset
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        client_accs = []

        if self.classes_names is not None:
            if getattr(self.cfg, "scenario", "class") == "domain":
                seen_class_names = self.classes_names
            else:
                class_ids_per_task = list(get_class_ids_per_task(self.cfg))
                seen_class_ids = []
                for t in range(self.task_id + 1):
                    seen_class_ids.extend(class_ids_per_task[t])
                seen_class_names = get_class_names(self.classes_names, seen_class_ids)

        else:
            seen_class_names = self.texts

        for client_id in range(self.num_clients):
            subsets = []
            for t in range(self.task_id + 1):
                task_dataset = self.eval_dataset[t:t + 1]
                dict_users_test = self.dict_users_test_all_tasks[t]
                client_indices = list(dict_users_test[client_id])
                if client_indices:
                    subsets.append(Subset(task_dataset, client_indices))

            concat_test = ConcatDataset(subsets)
            test_loader = DataLoader(concat_test, batch_size=self.cfg.batch_size, shuffle=False,
                                     num_workers=getattr(self.cfg, "num_workers", 1))

            local_model = copy.deepcopy(self.global_model).to(device)

            p_state = self.final_client_states[client_id]
            if p_state:
                p_state_on_device = {k: v.to(device) for k, v in p_state.items()}
                local_model.load_state_dict(p_state_on_device, strict=False)


            if self.classes_names is not None:
                try:
                    prev_ctx_tmp = local_model.prompt_learner.ctx.detach().clone()
                except AttributeError:
                    prev_ctx_tmp = None
                if hasattr(local_model, 'update_prompt_learner'):
                    local_model.update_prompt_learner(prev_ctx=prev_ctx_tmp, new_classnames=seen_class_names)

            local_model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels, _ in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    model_out = local_model(imgs)
                    outputs = model_out[0] if isinstance(model_out, tuple) else model_out
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = 100.0 * correct / total if total else 0.0
            client_accs.append(acc)

        mean_acc = sum(client_accs) / len(client_accs) if client_accs else 0.0

        print(f"[FedDuet] 任务 {self.task_id} | 平均准确率: {mean_acc:.2f}%")

        prev_accs = self.mean_acc_history
        total_tasks_done = len(prev_accs) + 1
        avg_acc = (sum(prev_accs) + mean_acc) / total_tasks_done

        self.mean_acc_history.append(mean_acc)

        import json, os
        dir_name = os.path.dirname(self.cfg.log_path)
        base = os.path.basename(self.cfg.log_path)
        path = os.path.join(dir_name, f"fedduet_client_{base}")

        log_entry = {
            "task": self.task_id,
            "acc": round(mean_acc, 2),
            "client_acc": [round(a, 2) for a in client_accs],
            "avg_acc": round(avg_acc, 2)
        }

        with open(path, 'a+') as f:
            f.write(json.dumps(log_entry) + '\n')

        print(f"[FedDuet] 任务 {self.task_id} | 累计平均准确率 (avg_acc): {avg_acc:.2f}%")

        return mean_acc

    def _update_prompt_pool(self):
        """全局Prompt池不直接通过梯度更新，而是通过聚合客户端上传的专家/prompt。此函数占位。"""
        return

    def _update_gate_network(self):
        """使用客户端上传的 <特征, 专家索引> 对来训练服务器端的门控网络。"""
        if not self._gate_train_buffer:
            return

        feats = torch.stack([f for f, _, _ in self._gate_train_buffer]).to(self.device)
        target_indices_list = [indices for _, indices, _ in self._gate_train_buffer]
        losses = torch.tensor([loss for _, _, loss in self._gate_train_buffer], device=self.device, dtype=torch.float)

        targets = torch.zeros(len(target_indices_list), self.prompt_pool_size, device=self.device)
        for i, indices in enumerate(target_indices_list):
            if isinstance(indices, int):
                indices = [indices]
            targets[i, indices] = 1.0

        eps = 1e-6
        weights = 1.0 / (losses + eps)
        weights = weights / weights.mean()

        logits = self.gate(feats.to(torch.float32))

        loss_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none').mean(dim=1)
        loss = (loss_bce * weights).mean()

        self.gate_optimizer.zero_grad()
        loss.backward()
        self.gate_optimizer.step()

        print(f"[Gate] 更新完成，loss={loss.item():.4f}, batch={feats.size(0)}")

        self._gate_train_buffer.clear()


def fedduet_train(global_model, train_dataset, eval_dataset, cfg, texts, task_id, client_model,
                             prev_ctx=None, prev_fusion_state=None, prev_mean_acc_history=None, classes_names=None,
                             prev_client_states=None):

    if hasattr(cfg, "seed"):
        seed = cfg.seed
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    print("\n=== FedDuet 训练配置 ===")
    print(f"communcation rounds: {getattr(cfg, 'com', 10)}")
    print(f"client epochs: {getattr(cfg, 'client_epochs', 5)}")
    print(f"LR: {getattr(cfg, 'lr', 3e-5)}")
    print(f"num clients: {getattr(cfg, 'num_clients', 5)}")
    print(f"IID: {'IID' if getattr(cfg, 'iid', True) else 'Non-IID'}")
    print(f"len(texts): {len(texts)}")

    wrapped_global_model = CustomCLIP(cfg, texts, global_model, prev_ctx, prev_fusion_state)
    wrapped_client_model = CustomCLIP(cfg, texts, client_model, prev_ctx, prev_fusion_state)
    
    trainer = FedDuetTrainer( # when initiating, the pool prompt is initiated.
        cfg=cfg,
        global_model=wrapped_global_model,
        client_model=wrapped_client_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        task_id=task_id,
        texts=texts,
        prev_ctx=prev_ctx,
        prev_fusion_state=prev_fusion_state,
        prev_mean_acc_history=prev_mean_acc_history,
        classes_names=classes_names,
        prev_client_states=prev_client_states
    )

    trained_model, ctx_state, fusion_state, mean_acc_history, final_client_states = trainer.train()

    return trained_model, ctx_state, fusion_state, mean_acc_history, final_client_states


class MultiheadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.2, scaling=1.0, dtype=torch.float32):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # self.scaling = self.embed_dim ** -0.5
        self.scaling = scaling
        self.dtype = dtype

        self.W_q = nn.Linear(d_model, d_model, dtype=self.dtype)
        self.W_k = nn.Linear(d_model, d_model, dtype=self.dtype)
        self.W_v = nn.Linear(d_model, d_model, dtype=self.dtype)
        self.W_o = nn.Linear(d_model, d_model, dtype=self.dtype)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output, attn_probs

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output, torch.mean(attn_probs, dim=1)

