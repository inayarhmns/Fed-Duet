import copy
import os
import json
from copy import deepcopy

import hydra
import logging
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

import torch
import statistics
from torch.utils.data import DataLoader
from continuum.metrics import Logger

from continual_clip import utils
from continual_clip.models import load_model
from continual_clip.datasets import build_cl_scenarios


def save_checkpoint(model, cfg, suffix="final"):
    """Save model checkpoint robustly.

    Args:
        model (torch.nn.Module): trained model to save.
        cfg (DictConfig): experiment configuration.
        suffix (str): filename suffix, e.g., 'final' or 'epoch_1'.
    Returns:
        str: path to the saved checkpoint.
    """
    try:
        # Prefer cfg.workdir if valid, else fallback to current directory.
        base_dir = getattr(cfg, "workdir", None)
        if not base_dir or not os.path.isdir(base_dir):
            logging.warning(f"cfg.workdir '{base_dir}' is invalid or does not exist. Saving checkpoint to current directory.")
            base_dir = os.getcwd()

        checkpoint_dir = os.path.join(base_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        ckpt_path = os.path.join(checkpoint_dir, f"{cfg.method}_{suffix}.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            ckpt_path,
        )
        print(f"Model checkpoint saved to {ckpt_path}")
        return ckpt_path
    except Exception as e:
        logging.exception("Failed to save model checkpoint.")
        return ""


@hydra.main(config_path=None, config_name=None, version_base="1.1")
def continual_clip(cfg: DictConfig) -> None:
    """Continual-CLIP entry point.
    
    Args:
        cfg (DictConfig): CIL experiment configuration.
    """
    cfg.workdir = utils.get_workdir(path=os.getcwd())
    cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)

    utils.set_seed(cfg.seed)
    # class_order is only used in class-incremental scenarios
    if cfg.scenario == "class":
        cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))
    else:
        # For other scenarios like domain-incremental, class_order is not needed.
        cfg.class_order = None



    utils.save_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model  = load_model(cfg, device)
    print("Get model")


    initial_model = deepcopy(model)
    initial_model.eval()

    
    eval_dataset, classes_names = build_cl_scenarios(   # returns CIL class and the classnames
        cfg, is_train=False, transforms=model.transforms
    )
    # print(eval_dataset, eval_dataset)
    print('eval_classname', classes_names)
    train_dataset, train_classes_names = build_cl_scenarios(
        cfg, is_train=True, transforms=model.transforms
    )
    print('train_classes_names', train_classes_names)
    model.classes_names = classes_names


    with open(cfg.log_path, 'w+') as f: 
        pass

    acc_list = []
    metric_logger = Logger(list_subsets=["test"])
    _old_network = None

    # test
    for task_id, _ in enumerate(eval_dataset):

        logging.info(f"Evaluation for task {task_id} has started.")

        #training
        model.adaptation(task_id, cfg, train_dataset, train_classes_names, _old_network, eval_dataset)
        _old_network = copy.deepcopy(model)
        _old_network.eval()

        if cfg.scenario == "class":
            # --- Class-Incremental Evaluation ---
            # For CIL, evaluate on all seen tasks combined.
            eval_loader = DataLoader(eval_dataset[:task_id + 1], batch_size=cfg.batch_size, shuffle=False, num_workers=8)
            
            for inputs, targets, task_ids in tqdm(eval_loader, desc=f"CIL Eval Task {task_id}"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(image=inputs, task_id=task_id)
                metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")
            
            current_accuracy = 100 * metric_logger.accuracy
            acc_list.append(current_accuracy)
            
            with open(cfg.log_path, 'a+') as f:
                f.write(json.dumps({
                    'task': task_id,
                    'acc': round(current_accuracy, 2),
                    'avg_acc': round(100 * metric_logger.average_incremental_accuracy, 2),
                    'forgetting': round(100 * metric_logger.forgetting, 6),
                    'acc_per_task': [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
                    'bwt': round(100 * metric_logger.backward_transfer, 2),
                    'fwt': round(100 * metric_logger.forward_transfer, 2),
                }) + '\n')
            metric_logger.end_task()

        elif cfg.scenario == "domain":
            # --- Domain-Incremental Evaluation ---
            # For DIL, evaluate on each seen domain individually and average the results.
            domain_accuracies = []
            for i in range(task_id + 1):
                domain_loader = DataLoader(eval_dataset[i], batch_size=cfg.batch_size, shuffle=False, num_workers=8)
                
                domain_correct = 0
                domain_total = 0
                with torch.no_grad():
                    for inputs, targets, task_ids in tqdm(domain_loader, desc=f"DIL Eval Domain {i}"):
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(image=inputs, task_id=task_id)
                        _, preds = torch.max(outputs.data, 1)
                        domain_total += targets.size(0)
                        domain_correct += (preds == targets).sum().item()
                
                domain_acc = 100.0 * domain_correct / domain_total if domain_total > 0 else 0
                domain_accuracies.append(domain_acc)
                logging.info(f"  - Accuracy on Domain {i}: {domain_acc:.2f}%")

            mean_accuracy = statistics.mean(domain_accuracies) if domain_accuracies else 0.0
            acc_list.append(mean_accuracy)

            with open(cfg.log_path, 'a+') as f:
                f.write(json.dumps({
                    'task': task_id,
                    'mean_acc': round(mean_accuracy, 2),
                    'domain_accs': [round(acc, 2) for acc in domain_accuracies]
                }) + '\n')
        else:
            raise ValueError(f"Unsupported scenario for evaluation: {cfg.scenario}")

    with open(cfg.log_path, 'a+') as f:
        if acc_list:
            f.write(json.dumps({
                'last': round(acc_list[-1], 2), 
                'avg': round(statistics.mean(acc_list), 2)
            }) + '\n')
        else:
            logging.warning("acc_list is empty. No tasks were evaluated or no results were recorded.")
            f.write(json.dumps({
                'last': 0, 
                'avg': 0,
                'error': 'acc_list is empty at the end of evaluation.'
            }) + '\n')

if __name__ == "__main__":
    continual_clip()