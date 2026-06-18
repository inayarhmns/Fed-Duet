
from omegaconf import DictConfig

import clip.clip as clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils import get_class_ids_per_task, get_class_names
import copy
from .FedDuet import fedduet_train


class ClassIncremental(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        # Load the standard CLIP model first
        self.model, self.transforms, _ = clip.load(cfg.model_name, device=device, jit=jit,cfg=cfg)

        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.cfg = cfg
        self.text_tokens = None #这是所有种类的text的tokens

        # ------- FedDuet 相关上下文 -------
        # 用于保存上一任务学习到的 prompt 上下文向量，首次任务为空
        self.ctx = None
        # 用于 FedDuet 保存客户端状态
        self.client_states = None
        # 用于 FedDuet 保存本地 fusion_gating 参数
        self.fusion_gating_state = None
        # --- 新增：保存个性化MoE专家状态 ---
        self.personalized_moe_expert_states = None
        # 用于保存跨任务的平均准确率历史
        self.mean_acc_history = None
        self.shared_state = {}






    def forward(self, image, text=None, task_id=0, is_train=False):
        if text is None:
            with torch.no_grad():
                    logits_per_image, aux_loss = self.model(image, None, task_id, is_train=False)
                    # 这里的probs需要处理连续学习中的类别偏移
                    probs = logits_per_image.softmax(dim=-1)
        else:
            with torch.no_grad():
                logits_per_image, _ = self.model(image, text, 0, is_train=False)
                probs = logits_per_image.softmax(dim=-1)
        return probs


    def get_task_classes(self, task_id):
        """Get the category name of the specified task ID"""
        return get_class_names(self.classes_names, self.class_ids_per_task[task_id])


    def adaptation(self, task_id, cfg, train_dataset, train_classes_names,_old_network = None,eval_dataset=None):
        
        # category name of the current task (not cumulative)
        current_task_class_names = self.get_task_classes(task_id) #these are the classes from the dataset, e.g. CIFAR10, the 10 tasks of them.
        print(f"Class-IL Task {task_id}: Training with classes {current_task_class_names}.")
        # Accumulation of all categories is only necessary during evaluation.
        if not hasattr(self, 'current_class_names'):
            self.current_class_names = []
        self.current_class_names += current_task_class_names

        # tokenize the class names
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        #TODO: whats this for?
        if cfg.method != "zeroshot":
            # update_prompt_learner is updating the CoOp style prompt vectors. 
            # instead of starting from the start "a bad photo of cat" 
            # it starts from the learned prompt vectors of the previous task and expands it to accommodate new classes. 
            # 对于fedduet，确保模型在训练前使用正确的上下文
            if hasattr(self.model, 'update_prompt_learner'): 
                # 训练时只使用当前任务的类别，但保持累积的类别列表以供评估使用
                    if task_id > 0 and hasattr(self, 'ctx') and self.ctx is not None:
                        self.model.update_prompt_learner(prev_ctx=self.ctx, new_classnames=current_task_class_names)
            # 执行训练
            self.clip_train(task_id, cfg, train_dataset,eval_dataset, train_classes_names, _old_network=_old_network)


    def clip_train(self, task_id, cfg, train_dataset,eval_dataset, train_classes_names, _old_network=None ):
        ### laoding dataset
        train_loader = DataLoader(train_dataset[task_id:task_id + 1],
                                  batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=8)


        train_iter = iter(train_loader)  # 获取每个step的数据集
        # print('cfg.batch_size',cfg.batch_size)


        EPOCH = 1
        num_batches = len(train_loader)
        total_iterations = EPOCH * num_batches

        # move model to device
        self.model = self.model.cuda()
        devices = list(range(torch.cuda.device_count()))
        print("Using devices", devices)
        print(f"[Task {task_id} START] GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")


        # texts only contains the number of classes of number of increments, e.g. 5
        # e.g. texts: ['a bad photo of a airplane.', 'a bad photo of a automobile.', 'a bad photo of a bird.', 'a bad photo of a cat.', 'a bad photo of a deer.']
        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        print(classnames)
        texts = [self.prompt_template.format(c) for c in classnames]
        print(f"texts: {texts}")
        #print("train_dataset class order:", train_classes_names)
        texts = clip.tokenize(texts).to(self.device)

        # All seen texts are only needed for evaluation in some methods, 
        # but training should use current task's texts.
        # all_seen_texts contains the tokenized texts of all classes seen so far, e.g. 50 classes after 10 increments with each 5 class per increment.

        all_seen_texts = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        print(f"all_seen_texts: {all_seen_texts}")
        # here the text's token are different. 49406 = opening token, 49407 = closing token, 320 = "a", 2103 = "bad", 1125 = "photo", 539 = "of", 16451 = "airplane", 25258 = "automobile"
        # all_seen_texts: tensor([[49406,   320,  2103,  1125,   539,   320, 16451,   269, 49407,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0],
        # [49406,   320,  2103,  1125,   539,   320, 25258,   269, 49407,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0],.. continue until it has increments dimension in the first dimension

        #TODO: If there are other methods, add here for training

        # start training
        self.model.train()
        if cfg.federated:#采用联邦学习
            print("----------------federated learning----------------")
            if cfg.use_FedDuet:
                # 调用 FedDuet 训练
                print("using FedDuet")
                print(f"[Task {task_id} BEFORE deepcopy] GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")

                import gc
                gc.collect()
                torch.cuda.empty_cache()
                global_model = self.model
                client_model = copy.deepcopy(self.model) 
                print(f"[Task {task_id} AFTER deepcopy] GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")

                current_classnames = self.get_task_classes(task_id)

                result = fedduet_train(
                    global_model=global_model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    cfg=cfg,
                    texts=current_classnames,
                    task_id=task_id,
                    client_model=client_model,
                    prev_ctx=self.ctx if hasattr(self, 'ctx') else None,
                    prev_fusion_state=self.fusion_gating_state if hasattr(self, 'fusion_gating_state') else None,
                    prev_mean_acc_history=getattr(self, 'mean_acc_history', None),
                    classes_names=self.classes_names,
                    prev_client_states=self.personalized_moe_expert_states if hasattr(self, 'personalized_moe_expert_states') else None
                )

                # Handle return values (model, ctx, fusion_state, mean_acc_history, personalized_states)
                if isinstance(result, tuple):
                    if len(result) == 5:
                        self.model, self.ctx, self.fusion_gating_state, self.mean_acc_history, self.personalized_moe_expert_states = result
                    elif len(result) == 4:
                        self.model, self.ctx, self.fusion_gating_state, self.mean_acc_history = result
                    elif len(result) == 3:
                        self.model, self.ctx, self.fusion_gating_state = result
                    else:
                        self.model, self.ctx = result
                else:
                    self.model = result
                    self.ctx = getattr(self, 'ctx', None)
                    self.fusion_gating_state = getattr(self, 'fusion_gating_state', None)
                    self.personalized_moe_expert_states = getattr(self, 'personalized_moe_expert_states', None)

                # After training, update the prompt learner to accommodate all seen classes
                if hasattr(self.model, 'update_prompt_learner') and hasattr(self, 'ctx'):
                    self.model.update_prompt_learner(prev_ctx=self.ctx,
                                                     new_classnames=self.current_class_names)
                else:
                    raise ValueError(f"Unsupported federated method in this refactored version. Only 'FedDuet' is supported.")

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.model = self.model.cuda()
        self.model.eval()





class DomainIncremental(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        # Load the standard CLIP model first
        self.model, self.transforms, _ = clip.load(cfg.model_name, device=device, jit=jit,cfg=cfg)

        self.current_class_names = []
        self.cfg = cfg
        self.text_tokens = None

        # To retain context across tasks
        self.ctx = None
        self.client_states = None
        self.fusion_gating_state = None
        self.personalized_moe_expert_states = None
        # To retain mean accuracy history across tasks
        self.mean_acc_history = None
        self.shared_state = {}

    def forward(self, image, text=None, task_id=0, is_train=False):
        if text is None:
            with torch.no_grad():
                    logits_per_image, aux_loss = self.model(image, None, task_id, is_train=False)
                    probs = logits_per_image.softmax(dim=-1)
        else:
            with torch.no_grad():
                logits_per_image, _ = self.model(image, text, 0, is_train=False)
                probs = logits_per_image.softmax(dim=-1)
        return probs


    def get_task_classes(self, task_id):
        # For Domain-IL, all classes are present in every task.
        return self.classes_names


    def adaptation(self, task_id, cfg, train_dataset, train_classes_names,_old_network = None,eval_dataset=None):
        # For Domain-IL, the class names are always the full set.
        self.current_class_names = self.classes_names

        # Update text_tokens for evaluation to cover all classes
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)

        print(f"Domain-IL Task {task_id}: Training with all {len(self.current_class_names)} classes.")

        # Call the training function
        self.clip_train(task_id, cfg, train_dataset,eval_dataset, train_classes_names, _old_network=_old_network)


    def clip_train(self, task_id, cfg, train_dataset, eval_dataset, train_classes_names, _old_network=None,
                   all_seen_texts=None):
        ### laoding dataset
        train_loader = DataLoader(train_dataset[task_id:task_id + 1],
                                  batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=8)
        train_iter = iter(train_loader)
        # print('cfg.batch_size',cfg.batch_size)

        EPOCH = 1
        num_batches = len(train_loader)
        total_iterations = EPOCH * num_batches

        self.model = self.model.cuda()

        # text
        # For Domain-IL, we use all class names for all tasks.
        classnames = self.classes_names
        texts = [self.prompt_template.format(c) for c in classnames]
        texts = clip.tokenize(texts).to(self.device)

        # TODO: If there are other methods, add here for training
        # start training
        self.model.train()
        if cfg.federated:
            print("----------------federated learning----------------")
            if cfg.use_fedDuet:
                print("using fedduet")
                print(f"[Task {task_id} BEFORE deepcopy] GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                global_model = self.model
                client_model = copy.deepcopy(self.model)
                print(f"[Task {task_id} AFTER deepcopy] GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")

                current_classnames = self.classes_names
                result = fedduet_train(
                    global_model=global_model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    cfg=cfg,
                    texts=current_classnames,
                    task_id=task_id,
                    client_model=client_model,
                    prev_ctx=self.ctx if hasattr(self, 'ctx') else None,
                    prev_fusion_state=self.fusion_gating_state if hasattr(self, 'fusion_gating_state') else None,
                    prev_mean_acc_history=getattr(self, 'mean_acc_history', None),
                    classes_names=self.classes_names,
                    prev_client_states=self.personalized_moe_expert_states if hasattr(self, 'personalized_moe_expert_states') else None,
                )

                if isinstance(result, tuple):
                    if len(result) == 5:
                        self.model, self.ctx, self.fusion_gating_state, self.mean_acc_history, self.personalized_moe_expert_states = result
                    elif len(result) == 4:
                        self.model, self.ctx, self.fusion_gating_state, self.mean_acc_history = result
                    elif len(result) == 3:
                        self.model, self.ctx, self.fusion_gating_state = result
                    else:
                        self.model, self.ctx = result
                else:
                    self.model = result
                    self.ctx = getattr(self, 'ctx', None)
                    self.fusion_gating_state = getattr(self, 'fusion_gating_state', None)
                    self.personalized_moe_expert_states = getattr(self, 'personalized_moe_expert_states', None)

                if hasattr(self.model, 'update_prompt_learner') and hasattr(self, 'ctx'):
                    self.model.update_prompt_learner(prev_ctx=self.ctx,
                                                     new_classnames=self.classes_names)

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.model = self.model.cuda()
        self.model.eval()

def load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.

    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.

    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        return ClassIncremental(cfg, device)
    elif cfg.scenario == "domain":
        return DomainIncremental(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario,
            Please choose from ['class', "domain', 'task-agnostic']
        """)
