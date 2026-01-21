# ===== train_kapt_correct.py (完整修复版) =====
import warnings

warnings.filterwarnings('ignore')

from argparse import Namespace
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from chemprop.train.run_training import run_training
from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs, save_checkpoint
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.torchlight import initialize_exp
from chemprop.models.model import build_model, KAPTPromptModule, PromptInjector


def load_pretrained_kano(args, num_tasks, logger=None):
    """从checkpoint加载预训练的KANO模型（兼容多种格式）"""
    info = logger.info if logger else print

    if args.checkpoint_path is None:
        raise ValueError('KAPT requires a pre-trained KANO checkpoint. Please specify --checkpoint_path.')

    info(f'Loading pre-trained KANO from: {args.checkpoint_path}')

    # 加载检查点文件
    if args.cuda:
        state = torch.load(args.checkpoint_path, map_location=f'cuda:{args.gpu}')
    else:
        state = torch.load(args.checkpoint_path, map_location='cpu')

    # ============ 兼容不同的检查点格式 ============
    if isinstance(state, dict):
        if 'state_dict' in state:
            # 格式 1: {'state_dict': ..., 'args': ..., ...}
            loaded_state_dict = state['state_dict']
            info('Loaded checkpoint format: full state dict')
        elif 'model_state_dict' in state:
            # 格式 2: {'model_state_dict': ..., ...}
            loaded_state_dict = state['model_state_dict']
            info('Loaded checkpoint format: model_state_dict')
        else:
            # 格式 3: 直接是 state_dict (没有嵌套)
            loaded_state_dict = state
            info('Loaded checkpoint format: direct state_dict')
    else:
        raise ValueError(f'Unexpected checkpoint format: {type(state)}')

    # ============ 设置 num_tasks 并构建模型 ============
    # 临时设置 num_tasks 用于构建模型
    args.num_tasks = num_tasks

    # 使用当前args构建模型
    model = build_model(args, encoder_name=args.encoder_name)

    # 加载预训练权重
    try:
        model.load_state_dict(loaded_state_dict, strict=False)
        info(f'Successfully loaded pre-trained KANO encoder')
    except Exception as e:
        info(f'Warning: Could not load all weights strictly. Error: {e}')
        # 尝试宽松加载
        missing_keys, unexpected_keys = model.load_state_dict(loaded_state_dict, strict=False)
        if missing_keys:
            info(f'Missing keys: {missing_keys[:5]}...' if len(missing_keys) > 5 else f'Missing keys: {missing_keys}')
        if unexpected_keys:
            info(f'Unexpected keys: {unexpected_keys[:5]}...' if len(
                unexpected_keys) > 5 else f'Unexpected keys: {unexpected_keys}')

    # 移动到设备
    if args.cuda:
        model = model.cuda(args.gpu)

    return model


def initialize_kapt_model(pretrained_kano, args, num_tasks, logger=None):
    """在预训练KANO基础上添加KAPT组件"""
    info = logger.info if logger else print
    device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')

    # 1. 使用预训练的KANO作为基础
    model = pretrained_kano

    # 2. 创建KAPT提示模块
    kapt_module = KAPTPromptModule(
        num_tasks=num_tasks,
        node_dim=args.hidden_size,
        hidden_dim=args.hidden_size,
        prompt_dim=getattr(args, 'prompt_dim', 128)
    ).to(device)

    # 3. 创建提示注入器
    prompt_injection_layers = getattr(args, 'prompt_injection_layers', list(range(args.depth)))
    prompt_injectors = nn.ModuleList([
        PromptInjector(args.hidden_size, getattr(args, 'prompt_dim', 128))
        for _ in prompt_injection_layers
    ]).to(device)

    # 4. 附加到模型
    model.kapt_module = kapt_module
    model.prompt_injectors = prompt_injectors
    model.use_kapt = True

    # 5. 日志输出
    info(f'KAPT module initialized:')
    info(f'  - Prompt dimension: {args.prompt_dim}')
    info(f'  - Injection layers: {prompt_injection_layers}')
    info(f'  - Number of tasks: {num_tasks}')

    # 6. 统计参数数量
    kano_params = sum(p.numel() for n, p in model.named_parameters()
                      if 'kapt_module' not in n and 'prompt_injectors' not in n)
    kapt_params = sum(p.numel() for n, p in model.named_parameters()
                      if 'kapt_module' in n or 'prompt_injectors' in n)
    total_params = kano_params + kapt_params

    info(f'  - KANO parameters: {kano_params:,} ({"frozen" if args.freeze_kano else "trainable"})')
    info(f'  - KAPT parameters: {kapt_params:,} (trainable)')
    info(f'  - Total parameters: {total_params:,}')

    return model


def build_kapt_optimizer(model, args, logger=None):
    """构建差异化学习率优化器"""
    info = logger.info if logger else print

    # 分离KANO和KAPT参数
    kano_params = []
    kapt_params = []

    for name, param in model.named_parameters():
        if 'kapt_module' in name or 'prompt_injectors' in name:
            kapt_params.append(param)
        else:
            # 如果freeze_kano=True,则不添加到优化器
            if not getattr(args, 'freeze_kano', False):
                kano_params.append(param)

    # 构建优化器
    if args.freeze_kano:
        optimizer = torch.optim.Adam(
            kapt_params,
            lr=args.prompt_lr,
            weight_decay=getattr(args, 'weight_decay', 0.0)
        )
        info(f'Optimizer: Prompt-only tuning (KANO frozen)')
        info(f'  - Prompt LR: {args.prompt_lr}')
    else:
        optimizer = torch.optim.Adam([
            {'params': kano_params, 'lr': args.kano_lr},
            {'params': kapt_params, 'lr': args.prompt_lr}
        ], weight_decay=getattr(args, 'weight_decay', 0.0))
        info(f'Optimizer: Differential learning rates')
        info(f'  - KANO LR: {args.kano_lr}')
        info(f'  - Prompt LR: {args.prompt_lr}')

    return optimizer


def run_kapt_training(args: Namespace, logger=None) -> Tuple[float, float, float, float]:
    """KAPT训练主逻辑"""
    info = logger.info if logger else print
    init_seed = args.seed
    save_dir = args.save_dir

    # ============ 先获取任务数量 ============
    task_names = get_task_names(args.data_path)
    num_tasks = len(task_names)
    args.num_tasks = num_tasks  # 设置到args中

    info(f'\n{"=" * 60}')
    info(f'KAPT Training Started')
    info(f'{"=" * 60}')
    info(f'Dataset: {os.path.basename(args.data_path)}')
    info(f'Number of tasks: {num_tasks}')
    info(f'Metric: {args.metric}')
    info(f'Pre-trained checkpoint: {args.checkpoint_path}')
    info(f'{"=" * 60}\n')

    # 存储结果
    all_val_scores = []
    all_test_scores = []
    best_models = []

    for run_num in range(args.num_runs):
        info(f'\n===== Run {run_num + 1}/{args.num_runs} =====')
        args.seed = init_seed + run_num
        args.save_dir = os.path.join(save_dir, f'run_{run_num}')
        makedirs(args.save_dir)

        # ============ Step 1: 加载预训练KANO ============
        pretrained_kano = load_pretrained_kano(args, num_tasks, logger)

        # ============ Step 2: 添加KAPT组件 ============
        if args.use_kapt:
            model = initialize_kapt_model(pretrained_kano, args, num_tasks, logger)
        else:
            model = pretrained_kano

        # ============ Step 3: 构建优化器 ============
        if args.use_kapt:
            optimizer = build_kapt_optimizer(model, args, logger)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)

        # ============ Step 4: 训练 ============
        # 将model和optimizer传递给run_training
        args.custom_model = model
        args.custom_optimizer = optimizer

        model_scores = run_training(
            args,
            prompt=args.use_kapt,
            logger=logger
        )

        # ============ Step 5: 处理结果 ============
        if isinstance(model_scores, list) and len(model_scores) >= 2:
            test_scores, val_scores = model_scores[0], model_scores[1]
        else:
            test_scores = val_scores = model_scores[0] if isinstance(model_scores, list) else model_scores

        val_score = np.nanmean(val_scores) if isinstance(val_scores, (list, np.ndarray)) else val_scores
        test_score = np.nanmean(test_scores) if isinstance(test_scores, (list, np.ndarray)) else test_scores

        all_val_scores.append(float(val_score))
        all_test_scores.append(float(test_score))

        info(f'Run {run_num + 1} | Val {args.metric}: {val_score:.4f} | Test {args.metric}: {test_score:.4f}')

        # 保存模型
        run_model_path = os.path.join(args.save_dir, 'model_0', 'model.pt')
        if os.path.exists(run_model_path):
            best_models.append(run_model_path)

    # ============ Step 6: 统计结果 ============
    all_val_scores = np.array(all_val_scores)
    all_test_scores = np.array(all_test_scores)

    mean_val = np.nanmean(all_val_scores)
    std_val = np.nanstd(all_val_scores)
    mean_test = np.nanmean(all_test_scores)
    std_test = np.nanstd(all_test_scores)

    info(f'\n{"=" * 60}')
    info(f'KAPT Training Results')
    info(f'{"=" * 60}')
    info(f'Validation {args.metric}: {mean_val:.6f} ± {std_val:.6f}')
    info(f'Test {args.metric}: {mean_test:.6f} ± {std_test:.6f}')
    info(f'{"=" * 60}\n')

    return mean_val, std_val, mean_test, std_test


def main():
    args = parse_train_args()

    # ============ KAPT专用参数设置 ============
    args.use_kapt = getattr(args, 'use_kapt', True)
    args.task_id = getattr(args, 'task_id', 0)
    args.prompt_dim = getattr(args, 'prompt_dim', 128)
    args.kano_lr = getattr(args, 'kano_lr', 1e-5)
    args.prompt_lr = getattr(args, 'prompt_lr', 1e-3)
    args.freeze_kano = getattr(args, 'freeze_kano', False)
    args.weight_decay = getattr(args, 'weight_decay', 0.0)

    # 设备配置
    args.cuda = args.gpu is not None and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    # 验证checkpoint_path
    if args.use_kapt and args.checkpoint_path is None:
        raise ValueError(
            'KAPT requires a pre-trained KANO checkpoint!\n'
            'Please specify --checkpoint_path /path/to/pretrained_KANO.pkl'
        )

    modify_train_args(args)
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))

    # 执行KAPT训练
    mean_val, std_val, mean_test, std_test = run_kapt_training(args, logger)

    # 最终输出
    logger.info(f'\n🎉 KAPT Training Completed Successfully!')
    logger.info(f'Final Results:')
    logger.info(f'  Validation: {mean_val:.5f} ± {std_val:.5f}')
    logger.info(f'  Test: {mean_test:.5f} ± {std_test:.5f}')

    print(f'\n🎉 KAPT Training Completed!')
    print(f'Validation: {mean_val:.5f} ± {std_val:.5f}')
    print(f'Test: {mean_test:.5f} ± {std_test:.5f}')


if __name__ == '__main__':
    main()
