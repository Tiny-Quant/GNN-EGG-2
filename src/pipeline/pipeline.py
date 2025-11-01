"""
Experiment pipeline runner for graph-level explainability experiments.

Pipeline stages:
  1. Train / load explainee model (graph classifier)
  2. Train / load distance model (e.g. SimGNN)  [optional right now]
  3. Compute per-class mean embeddings from explainee
  4. Train a class-conditional generator for each target class

Usage:
  python -m src.pipeline.pipeline --config config/mutag.yaml
  python -m src.pipeline.pipeline --config config/proteins.yaml
  python -m src.pipeline.pipeline --config config/mutag.yaml --dry-run
"""

import argparse
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from src.trainers.train_explainee import train_explainee
from src.trainers.train_distance import train_distance
from src.trainers.train_generator import train_generator
from src.models.explainee_gnn import ExplaineeGIN
from src.models.adapter import GeneratorAdapter
from src.utils.embeddings import compute_classwise_means
from src.models.sim_gnn import SimGNN


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    def coerce_value(v):
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            try:
                if any(c in v for c in [".", "e", "E"]):
                    return float(v)
                return int(v)
            except ValueError:
                return v
        return v

    for section in ("explainee", "distance_model", "generator"):
        if section not in cfg:
            continue
        cfg[section] = {k: coerce_value(v) for k, v in cfg[section].items()}

    return cfg


def print_config_summary(cfg):
    print("\n🧾 Configuration Summary")
    print("──────────────────────────────")
    print(f"Experiment: {cfg['experiment']['name']}")
    print(f"Device: {cfg['experiment']['device']}")
    print(f"Seed: {cfg['experiment']['seed']}")
    print(f"Dataset: {cfg['dataset']['name']}")
    print(f"Output Directory: {cfg['experiment']['output_dir']}\n")

    for section in ["explainee", "distance_model", "generator"]:
        if section not in cfg:
            continue
        print(f"[{section.upper()}]")
        for k, v in cfg[section].items():
            print(f"  {k}: {v}")
        print("")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run graph explainability experiment pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/mutag.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview configuration and exit without training.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain models even if checkpoints already exist.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["experiment"]["device"])
    set_seed(cfg["experiment"]["seed"])
    Path(cfg["experiment"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Starting experiment: {cfg['experiment']['name']}")
    print(f"📄 Config loaded from: {args.config}")
    print(f"🖥️  Using device: {device}\n")

    if args.dry_run:
        print_config_summary(cfg)
        print("✅ Dry-run complete. No models were trained.\n")
        return

    # -----------------------------------------------------------------
    # (1) Train or load explainee model
    # -----------------------------------------------------------------
    print("🧠 Stage 1: Explainee model...")

    explainee_cfg = cfg["explainee"]
    explainee_ckpt = Path(explainee_cfg["save_path"])

    dataset_name = cfg["dataset"]["name"].lower()
    if dataset_name == "mutag":
        from src.datasets.mutag import load_mutag as load_dataset
        num_classes = 2
    elif dataset_name == "proteins":
        from src.datasets.proteins import load_proteins as load_dataset
        num_classes = 2
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' not supported yet.")

    dataset = load_dataset()

    if explainee_ckpt.exists() and not args.retrain:
        print(f"📂 Found explainee checkpoint at {explainee_ckpt}, loading...")
        in_dim = dataset[0].x.size(1)
        model_explainee = ExplaineeGIN(
            in_dim=in_dim,
            hidden_dim=explainee_cfg["hidden_dim"],
            num_layers=explainee_cfg["num_layers"],
            dropout=explainee_cfg["dropout"],
            num_classes=num_classes,
        ).to(device)
        state_dict = torch.load(explainee_ckpt, map_location=device, weights_only=False)
        model_explainee.load_state_dict(state_dict)
        model_explainee.eval()
    else:
        print("🛠 Training explainee from scratch...")
        model_explainee, _acc = train_explainee(
            dataset_name=cfg["dataset"]["name"],
            hidden_dim=explainee_cfg["hidden_dim"],
            num_layers=explainee_cfg["num_layers"],
            dropout=explainee_cfg["dropout"],
            lr=explainee_cfg["lr"],
            batch_size=explainee_cfg["batch_size"],
            epochs=explainee_cfg["epochs"],
            save_path=explainee_cfg["save_path"],
            checkpoint_dir=explainee_cfg["checkpoint_dir"],
            device=device,
        )
        model_explainee = model_explainee.to(device).eval()

    # -----------------------------------------------------------------
    # (2) Train distance model (optional / SimGNN)
    # -----------------------------------------------------------------
    print("\n📏 Stage 2: Distance model...")
    distance_cfg = cfg["distance_model"]
    distance_ckpt = Path(distance_cfg["save_path"])
    model_name = distance_cfg["model_name"].lower()

    distance_models: Dict[str, torch.nn.Module] = {}
    if model_name == "simgnn":
        simgnn_model: Optional[SimGNN] = None
        in_dim = dataset[0].x.size(1)
        if distance_ckpt.exists() and not args.retrain:
            print(f"📂 Found SimGNN checkpoint at {distance_ckpt}, loading...\n")
            simgnn_model = SimGNN(
                in_dim,
                hidden_dim=distance_cfg["hidden_dim"],
                use_tensor=distance_cfg.get("use_tensor", False),
                tensor_channels=distance_cfg.get("tensor_channels", 8),
            )
            state = torch.load(distance_ckpt, map_location=device, weights_only=False)
            simgnn_model.load_state_dict(state)
        else:
            simgnn_model, _ = train_distance(
                dataset_name=cfg["dataset"]["name"],
                data_path=distance_cfg["data_path"],
                hidden_dim=distance_cfg["hidden_dim"],
                lr=distance_cfg["lr"],
                batch_size=distance_cfg["batch_size"],
                epochs=distance_cfg["epochs"],
                use_tensor=distance_cfg["use_tensor"],
                alpha=distance_cfg["alpha"],
                save_path=distance_cfg["save_path"],
                checkpoint_dir=distance_cfg["checkpoint_dir"],
                device=device,
                overwrite_data=False,
            )

        if simgnn_model is not None:
            simgnn_model = simgnn_model.to(device).eval()
            for param in simgnn_model.parameters():
                param.requires_grad_(False)
            distance_models["simgnn"] = simgnn_model
    else:
        print(f"⚡ Skipping distance model stage (model_name={model_name}).\n")

    # -----------------------------------------------------------------
    # (3) Compute classwise means for embedding supervision
    # -----------------------------------------------------------------
    print("📊 Stage 3: Computing classwise embedding means...")

    # small loader across the WHOLE dataset for means
    train_data, _ = train_test_split(dataset, test_size=0.2, random_state=42)
    mean_loader = DataLoader(train_data, batch_size=explainee_cfg["batch_size"], shuffle=False)

    classwise_means = compute_classwise_means(
        model=model_explainee,
        dataloader=mean_loader,
        device=device,
        num_classes=num_classes,
        layer_names=None,  # auto-discover
    )
    print(f"🧭 Got per-class means for {len(classwise_means)} layers.")

    # -----------------------------------------------------------------
    # (4) Train separate generator for each class
    # -----------------------------------------------------------------
    print("\n🎨 Stage 4: Training generators per class...")

    # describe real dataset feature structure
    adapter = GeneratorAdapter(dataset)
    print(
        f"[Adapter spec] max_nodes={adapter.max_nodes}, "
        f"node blocks={adapter.dis_node_blocks}, edge blocks={adapter.dis_edge_blocks}, "
        f"cont node={adapter.num_cont_node_feats}, cont edge={adapter.num_cont_edge_feats}"
    )

    gen_cfg = cfg.get("generator", {})
    distance_gen_cfg = gen_cfg.get("distance", {})
    lambda_distance = gen_cfg.get("lambda_distance", distance_gen_cfg.get("lambda", 1.0))
    distance_metrics = distance_gen_cfg.get("metrics", [])
    distance_thresh = distance_gen_cfg.get("thresh", 0.5)

    filtered_metrics = []
    for spec in distance_metrics:
        name = str(spec.get("name", "")).lower()
        if name == "simgnn" and "simgnn" not in distance_models:
            print("⚠️ Distance metric 'simgnn' requested but SimGNN model is unavailable. Skipping this term.")
            continue
        filtered_metrics.append(spec)
    distance_metrics = filtered_metrics

    class_graphs = defaultdict(list)
    for graph in dataset:
        label = int(graph.y.item()) if hasattr(graph, "y") else 0
        class_graphs[label].append(graph)

    for class_idx in range(num_classes):
        ckpt_path = Path(gen_cfg["save_path"]).parent / f"generator_class{class_idx}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        if ckpt_path.exists() and not args.retrain:
            print(f"📂 Generator for class {class_idx} already exists → {ckpt_path}")
            continue

        print(f"\n🧬 Training Generator → class={class_idx}")
        train_generator(
            explainee=model_explainee,
            num_classes=num_classes,
            classwise_means=classwise_means,
            target_class=class_idx,
            save_path=str(ckpt_path),
            device=device,
            # arch spec from adapter
            max_nodes=adapter.max_nodes,
            num_cont_node_feats=adapter.num_cont_node_feats,
            dis_node_blocks=adapter.dis_node_blocks,
            num_cont_edge_feats=adapter.num_cont_edge_feats,
            dis_edge_blocks=adapter.dis_edge_blocks,
            # hparams
            lr=gen_cfg.get("lr", 1e-3),
            steps=gen_cfg.get("steps", 200),
            batch_size=gen_cfg.get("batch_size", 1),
            temperature=gen_cfg.get("temperature", 1.0),
            symmetric_adj=gen_cfg.get("symmetric_adj", True),
            allow_self_loops=gen_cfg.get("allow_self_loops", False),
            lambda_pull=gen_cfg.get("lambda_pull", 1.0),
            lambda_push=gen_cfg.get("lambda_push", 1.0),
            lambda_edge=gen_cfg.get("lambda_edge", 0.1),
            lambda_pred=gen_cfg.get("lambda_pred", 1.0),
            lambda_distance=lambda_distance,
            distance_metrics=distance_metrics,
            distance_models=distance_models,
            class_graphs=dict(class_graphs),
            distance_thresh=distance_thresh,
            log_every=gen_cfg.get("log_every", 20),
        )

    print("\n✅ Experiment complete!\n")


if __name__ == "__main__":
    main()
