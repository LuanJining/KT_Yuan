"""
LARESKT V7.2 超参数调优脚本

支持两种调优策略:
1. Optuna (推荐) - 贝叶斯优化，用少量试验找到最优参数
2. Grid Search - 穷举网格搜索

用法:
  # Optuna (默认，30次试验)
  python hyperparam_tune_v7_2.py --dataset_name assist2015 --n_trials 30

  # 指定 GPU 和试验次数
  python hyperparam_tune_v7_2.py --dataset_name assist2015 --n_trials 50 --gpu_ids 0

  # Grid Search
  python hyperparam_tune_v7_2.py --dataset_name assist2015 --method grid --n_trials 100

  # 恢复中断的 Optuna study
  python hyperparam_tune_v7_2.py --dataset_name assist2015 --study_name my_study --resume

  # 指定优化目标: validauc, validacc, testauc
  python hyperparam_tune_v7_2.py --dataset_name assist2015 --optimize_metric validauc --n_trials 30

  # 多 fold 交叉验证
  python hyperparam_tune_v7_2.py --dataset_name assist2015 --folds 0 1 2 --n_trials 30
"""

import argparse
import os
import sys
import json
import copy
import math
import random
import uuid
import hashlib
import itertools
from datetime import datetime

import torch
torch.set_num_threads(4)

# 将项目根目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pykt.models import train_model, init_model
from pykt.utils import set_seed
from pykt.datasets import init_dataset4train


# ============================================================
# 默认超参数 (作为 baseline)
# ============================================================
DEFAULT_PARAMS = {
    "model_name": "lareskt_v7_2",
    "emb_type": "qid",
    "save_dir": "saved_model_tune",
    "seed": 42,
    "dropout": 0.1,
    "d_model": 256,
    "d_ff": 256,
    "num_attn_heads": 4,
    "n_blocks": 1,
    "learning_rate": 1e-4,
    "batch_size": 64,
    "num_epochs": 100,
    "mean_recurrence": 3,
    "sampling_scheme": "uniform",
    "state_init_method": "zero",
    "state_std": 0.02,
    "state_scale": 1.0,
    "adapter_type": "linear",
    "tau": 0.2,
    "alpha": 0.0,
    "gamma": 0.0,
    "same_step": 1,
    "query_fusion": "gate",
    "use_wandb": 0,
    "add_uuid": 0,
    "gpu_ids": "",
}

# Optuna 搜索空间定义
SEARCH_SPACE = {
    "learning_rate": ("log", 1e-5, 1e-3),
    "d_model": ("choice", [128, 256, 512]),
    "d_ff": ("choice", [128, 256, 512]),
    "n_blocks": ("int", 1, 4),
    "num_attn_heads": ("choice", [2, 4, 8]),
    "dropout": ("float", 0.05, 0.3),
    "batch_size": ("choice", [32, 64, 128]),
    "mean_recurrence": ("int", 1, 6),
    "tau": ("float", 0.05, 0.5),
    "state_std": ("log", 0.001, 0.1),
    "state_scale": ("float", 0.5, 2.0),
}

CATEGORICAL_PARAMS = {
    "sampling_scheme": ["uniform", "fixed"],
    "adapter_type": ["linear", "concat"],
    "query_fusion": ["gate", "add", "concat"],
}

# Grid Search 候选值
GRID_SPACE = {
    "learning_rate": [5e-5, 1e-4, 3e-4],
    "d_model": [128, 256, 512],
    "d_ff": [256, 512],
    "n_blocks": [1, 2, 3],
    "dropout": [0.1, 0.15, 0.2],
    "batch_size": [32, 64],
    "mean_recurrence": [2, 3, 5],
    "tau": [0.1, 0.2, 0.3],
}


def parse_gpu_ids(gpu_ids):
    if gpu_ids is None or str(gpu_ids).strip() == "":
        return ""
    return str(gpu_ids).strip()


def sample_params_optuna(trial):
    """使用 Optuna trial 采样一组超参数"""
    params = copy.deepcopy(DEFAULT_PARAMS)
    for name, (method, *args) in SEARCH_SPACE.items():
        if method == "log":
            params[name] = trial.suggest_float(name, args[0], args[1], log=True)
        elif method == "float":
            params[name] = trial.suggest_float(name, args[0], args[1])
        elif method == "int":
            params[name] = trial.suggest_int(name, args[0], args[1])
        elif method == "choice":
            params[name] = trial.suggest_categorical(name, args[0])

    for name, choices in CATEGORICAL_PARAMS.items():
        params[name] = trial.suggest_categorical(name, choices)

    return params


def sample_params_random():
    """纯随机采样 (不依赖 Optuna)"""
    params = copy.deepcopy(DEFAULT_PARAMS)
    for name, (method, *args) in SEARCH_SPACE.items():
        if method == "log":
            log_val = random.uniform(math.log(args[0]), math.log(args[1]))
            params[name] = math.exp(log_val)
        elif method == "float":
            params[name] = random.uniform(args[0], args[1])
        elif method == "int":
            params[name] = random.randint(args[0], args[1])
        elif method == "choice":
            params[name] = random.choice(args[0])

    for name, choices in CATEGORICAL_PARAMS.items():
        params[name] = random.choice(choices)

    return params


def generate_grid_combinations():
    """生成所有网格搜索组合"""
    keys = list(GRID_SPACE.keys())
    values = list(GRID_SPACE.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def run_single_trial(params, dataset_name, fold, optimize_metric="validauc"):
    """运行单次试验，返回指定指标值"""

    # 确保 d_ff >= d_model 且能被 num_attn_heads 整除
    if params["d_ff"] < params["d_model"]:
        params["d_ff"] = params["d_model"]
    if params["d_model"] % params["num_attn_heads"] != 0:
        # 找到能整除的最大头数
        for h in [params["num_attn_heads"] - 1, params["num_attn_heads"] + 1,
                   params["num_attn_heads"] // 2]:
            if h > 0 and params["d_model"] % h == 0:
                params["num_attn_heads"] = h
                break

    set_seed(params["seed"])

    model_name = params["model_name"]
    emb_type = params["emb_type"]
    save_dir = params["save_dir"]
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    learning_rate = params["learning_rate"]

    # 加载配置
    with open("../configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        train_config["batch_size"] = batch_size
        train_config["num_epochs"] = num_epochs

    model_config = copy.deepcopy(params)
    for key in ["model_name", "dataset_name", "emb_type", "save_dir", "fold",
                "seed", "gpu_ids", "batch_size", "num_epochs"]:
        if key in model_config:
            del model_config[key]

    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)

    if "maxlen" in data_config[dataset_name]:
        train_config["seq_len"] = data_config[dataset_name]["maxlen"]

    seq_len = train_config["seq_len"]

    # 初始化数据
    train_loader, valid_loader, *_ = init_dataset4train(
        dataset_name, model_name, data_config, fold, batch_size, args=params
    )

    # 构建唯一路径 (用参数的 hash 值，避免文件名过长)
    params_snapshot = json.dumps({k: v for k, v in sorted(params.items())
                                   if k not in ["other_config", "gpu_ids"]}, sort_keys=True)
    params_hash = hashlib.md5(params_snapshot.encode()).hexdigest()[:10]
    trial_id = f"{dataset_name}_fold{fold}_{params_hash}"
    ckpt_path = os.path.join(save_dir, trial_id)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)

    # 在 ckpt_path 中保存完整参数以便复现
    with open(os.path.join(ckpt_path, "tune_params.json"), "w") as f:
        json.dump(params, f, indent=2, default=str)

    # 构建完整 model_config (和 wandb_train.py 保持一致)
    model_config_final = copy.deepcopy(model_config)
    if model_name in ["saint", "saint++", "sakt", "atdkt", "simplekt",
                       "stablekt", "datakt", "folibikt", "lareskt", "lareskt_v7_2"]:
        model_config_final["seq_len"] = seq_len

    # 初始化模型
    model = init_model(model_name, model_config_final, data_config[dataset_name], emb_type)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    opt = torch.optim.Adam(model.parameters(), learning_rate)

    # 训练
    testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = \
        train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path,
                    None, None, save_model=False)

    # 返回目标指标
    metrics = {
        "validauc": validauc,
        "validacc": validacc,
        "testauc": testauc,
        "testacc": testacc,
        "window_testauc": window_testauc,
        "best_epoch": best_epoch,
    }

    return metrics, params


def run_optuna(args):
    """使用 Optuna 进行贝叶斯超参数优化"""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.INFO)

    study_name = args.study_name or f"lareskt_v7_2_{args.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage_path = f"sqlite:///tune_results/{study_name}.db"
    os.makedirs("tune_results", exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=42)
    if args.resume:
        study = optuna.load_study(study_name=study_name, storage=storage_path)
        print(f"Resumed study: {study_name},已有 {len(study.trials)} 次试验")
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction="maximize",
            sampler=sampler,
        )

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    def objective(trial):
        params = sample_params_optuna(trial)
        params["dataset_name"] = args.dataset_name
        params["gpu_ids"] = args.gpu_ids

        # 多 fold 取平均
        fold_metrics = []
        for fold in args.folds:
            trial_params = copy.deepcopy(params)
            trial_params["fold"] = fold
            try:
                metrics, _ = run_single_trial(trial_params, args.dataset_name, fold, args.optimize_metric)
                fold_metrics.append(metrics)
                # 记录所有指标到 trial
                for k, v in metrics.items():
                    trial.set_user_attr(f"{k}_fold{fold}", v)
            except Exception as e:
                print(f"Fold {fold} 失败: {e}")
                return 0.0

        # 多 fold 平均
        avg_metrics = {}
        for k in fold_metrics[0].keys():
            avg_metrics[k] = sum(m[k] for m in fold_metrics) / len(fold_metrics)

        target = avg_metrics[args.optimize_metric]
        for k, v in avg_metrics.items():
            trial.set_user_attr(f"avg_{k}", v)

        print(f"\n{'='*60}")
        print(f"Trial {trial.number} | {args.optimize_metric} = {target:.4f}")
        for k, v in avg_metrics.items():
            print(f"  avg_{k}: {v:.4f}")
        print(f"{'='*60}\n")

        return target

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    # 输出结果
    print("\n" + "=" * 60)
    print("调优完成!")
    print(f"最佳 {args.optimize_metric}: {study.best_value:.4f}")
    print(f"最佳参数:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 保存结果
    result = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_user_attrs": dict(study.best_trial.user_attrs),
        "n_trials": len(study.trials),
        "dataset": args.dataset_name,
        "optimize_metric": args.optimize_metric,
        "folds": args.folds,
        "timestamp": datetime.now().isoformat(),
    }
    result_path = f"tune_results/{study_name}_best.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"结果已保存到: {result_path}")

    return study


def run_grid(args):
    """网格搜索"""
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    os.makedirs("tune_results", exist_ok=True)
    results = []
    total = 0

    # 预计算总数
    for _ in generate_grid_combinations():
        total += 1
    total *= len(args.folds)

    print(f"网格搜索: {total} 组合 (参数组合 x {len(args.folds)} folds)")

    count = 0
    for combo in generate_grid_combinations():
        params = copy.deepcopy(DEFAULT_PARAMS)
        params.update(combo)
        params["dataset_name"] = args.dataset_name
        params["gpu_ids"] = args.gpu_ids

        fold_metrics = []
        for fold in args.folds:
            trial_params = copy.deepcopy(params)
            trial_params["fold"] = fold
            try:
                metrics, _ = run_single_trial(trial_params, args.dataset_name, fold, args.optimize_metric)
                fold_metrics.append(metrics)
            except Exception as e:
                print(f"参数 {combo} Fold {fold} 失败: {e}")
                continue

            count += 1
            print(f"[{count}/{total}] Fold {fold} | {args.optimize_metric} = {metrics[args.optimize_metric]:.4f} | params = {combo}")

        if fold_metrics:
            avg_metrics = {}
            for k in fold_metrics[0].keys():
                avg_metrics[k] = sum(m[k] for m in fold_metrics) / len(fold_metrics)

            result = {"params": combo, "avg_metrics": avg_metrics, "fold_metrics": fold_metrics}
            results.append(result)

    # 按目标指标排序
    results.sort(key=lambda x: x["avg_metrics"][args.optimize_metric], reverse=True)

    print("\n" + "=" * 60)
    print("网格搜索完成! Top 5:")
    for i, r in enumerate(results[:5]):
        print(f"\n#{i+1} | {args.optimize_metric} = {r['avg_metrics'][args.optimize_metric]:.4f}")
        print(f"  params: {r['params']}")
        for k, v in r["avg_metrics"].items():
            print(f"  avg_{k}: {v:.4f}")

    # 保存结果
    result_path = f"tune_results/grid_{args.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到: {result_path}")


def run_random(args):
    """随机搜索"""
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    os.makedirs("tune_results", exist_ok=True)
    results = []

    print(f"随机搜索: {args.n_trials} 次试验, {len(args.folds)} folds")

    for trial_idx in range(args.n_trials):
        params = sample_params_random()
        params["dataset_name"] = args.dataset_name
        params["gpu_ids"] = args.gpu_ids

        fold_metrics = []
        for fold in args.folds:
            trial_params = copy.deepcopy(params)
            trial_params["fold"] = fold
            try:
                metrics, _ = run_single_trial(trial_params, args.dataset_name, fold, args.optimize_metric)
                fold_metrics.append(metrics)
            except Exception as e:
                print(f"Trial {trial_idx} Fold {fold} 失败: {e}")
                continue

        if fold_metrics:
            avg_metrics = {}
            for k in fold_metrics[0].keys():
                avg_metrics[k] = sum(m[k] for m in fold_metrics) / len(fold_metrics)

            result = {
                "trial": trial_idx,
                "params": {k: v for k, v in sorted(params.items())},
                "avg_metrics": avg_metrics,
                "fold_metrics": fold_metrics,
            }
            results.append(result)

            print(f"[{trial_idx+1}/{args.n_trials}] {args.optimize_metric} = "
                  f"{avg_metrics[args.optimize_metric]:.4f} | lr={params['learning_rate']:.6f} "
                  f"d_model={params['d_model']} n_blocks={params['n_blocks']} "
                  f"dropout={params['dropout']:.3f} mr={params['mean_recurrence']}")

    # 排序
    results.sort(key=lambda x: x["avg_metrics"][args.optimize_metric], reverse=True)

    print("\n" + "=" * 60)
    print("随机搜索完成! Top 5:")
    for i, r in enumerate(results[:5]):
        print(f"\n#{i+1} | {args.optimize_metric} = {r['avg_metrics'][args.optimize_metric]:.4f}")
        for k, v in r["params"].items():
            print(f"  {k}: {v}")

    # 保存
    result_path = f"tune_results/random_{args.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到: {result_path}")


def main():
    parser = argparse.ArgumentParser(description="LARESKT V7.2 超参数调优")

    # 基本设置
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--method", type=str, default="optuna",
                        choices=["optuna", "grid", "random"],
                        help="调优方法: optuna (贝叶斯), grid (网格), random (随机)")
    parser.add_argument("--optimize_metric", type=str, default="validauc",
                        choices=["validauc", "validacc", "testauc", "testacc"],
                        help="优化目标指标")
    parser.add_argument("--gpu_ids", type=str, default="")

    # 试验控制
    parser.add_argument("--n_trials", type=int, default=30, help="试验次数 (optuna/random)")
    parser.add_argument("--timeout", type=int, default=None, help="超时时间(秒) (optuna)")
    parser.add_argument("--folds", type=int, nargs="+", default=[0], help="使用的 fold")

    # 模型固定参数 (不参与搜索)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emb_type", type=str, default="qid")

    # Optuna 设置
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study 名称")
    parser.add_argument("--resume", action="store_true", help="恢复已有的 Optuna study")

    args = parser.parse_args()

    # 更新默认参数
    DEFAULT_PARAMS["num_epochs"] = args.num_epochs
    DEFAULT_PARAMS["seed"] = args.seed
    DEFAULT_PARAMS["emb_type"] = args.emb_type

    print("=" * 60)
    print(f"LARESKT V7.2 超参数调优")
    print(f"  数据集: {args.dataset_name}")
    print(f"  方法: {args.method}")
    print(f"  优化目标: {args.optimize_metric}")
    print(f"  Folds: {args.folds}")
    print(f"  试验次数: {args.n_trials if args.method != 'grid' else '网格搜索全部组合'}")
    print("=" * 60)

    if args.method == "optuna":
        run_optuna(args)
    elif args.method == "grid":
        run_grid(args)
    elif args.method == "random":
        run_random(args)


if __name__ == "__main__":
    main()
