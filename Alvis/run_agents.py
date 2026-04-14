import os
os.environ["WANDB_START_METHOD"] = "thread"

import sys
import argparse
import wandb
import yaml
import torch
import multiprocessing as mp

from itertools import product
from joblib import Parallel, delayed

from models.model import inject_rolora_to_llama, get_model


# ==========================================================
# CONFIG HELPERS
# ==========================================================

def load_config_from_yaml(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] Config not found: {filepath}")
        return None
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def nest_dot_keys(config_dict):
    def insert_nested(d, keys, value):
        if len(keys) == 1:
            d[keys[0]] = value
        else:
            if keys[0] not in d:
                d[keys[0]] = {}
            insert_nested(d[keys[0]], keys[1:], value)
    nested = {}
    for key, value in config_dict.items():
        insert_nested(nested, key.split("."), value)
    return nested


def generate_runs_from_sweep_config(sweep_config, num_run_per_config):
    params       = sweep_config.get("parameters", {})
    param_names  = list(params.keys())
    param_values = [param["values"] for param in params.values()]
    combinations = list(product(*param_values))
    runs = [
        {k: v for k, v in zip(param_names, combo)}
        for combo in combinations
    ]
    return runs * num_run_per_config


# ==========================================================
# HELPER: apply RoLoRA + memory fixes for LLaMA
# ==========================================================

def prepare_llama_model(model, rank):
    """
    Injects RoLoRA adapters and applies memory optimisations required
    to fit LLaMA3-8B training inside 32 GB VRAM.
    """
    model = inject_rolora_to_llama(model, rank=rank)

    # Recomputes activations during backward instead of storing them.
    # Saves ~40% activation memory at ~20% extra compute cost.
    print("[ACTION] Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()

    # KV cache is incompatible with gradient checkpointing — must be off.
    model.config.use_cache = False

    return model


# ==========================================================
# TRAIN FUNCTION
# ==========================================================

def train(config, model):
    from server.server_sparse import SparseFLServer
    from server.server_fedavg import FedAvgServer

    c = config if hasattr(config, "get") else config.as_dict()

    dataset_name        = c.get("dataset_name", "MNIST")
    num_clients         = c.get("num_clients", 10)
    fraction_malicious  = c.get("fraction_malicious", 0.0)
    total_epochs        = c.get("total_epochs", 10)
    q_factor            = c.get("q_factor", 0.1)
    evaluate_each_epoch = c.get("evaluate_each_epoch", 1)
    attack_args         = c.get("attack_args", None)
    defence_args        = c.get("defence_args", None)
    aggregate_type      = c.get("aggregate_type", "fedavg")
    batch_size          = c.get("batch_size", 64)
    local_epochs        = c.get("local_epochs", 1)
    malicious_type      = c.get("malicious_type", "group_oriented")
    normalize_params    = c.get("normalize_params", False)
    multi_attack_args   = c.get("multi_attack_args", None)
    learning_rate       = c.get("learning_rate", 1e-4)   # ✅ read and pass explicitly

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[DEBUG] Device        : {device}")
    print(f"[DEBUG] Learning rate : {learning_rate}")
    print(f"[DEBUG] Attack        : {attack_args}")

    server_args = {
        "dataset_name":        dataset_name,
        "num_clients":         num_clients,
        "fraction_malicious":  fraction_malicious,
        "attack_args":         attack_args,
        "defence_args":        defence_args,
        "total_epochs":        total_epochs,
        "q_factor":            q_factor,
        "model":               model,
        "evaluate_each_epoch": evaluate_each_epoch,
        "batch_size":          batch_size,
        "local_epochs":        local_epochs,
        "malicious_type":      malicious_type,
        "device":              device,
        "normalize_params":    normalize_params,
        "multi_attack_args":   multi_attack_args,
        "learning_rate":       learning_rate,   # ✅ passed to server → clients
    }

    if aggregate_type == "sparse":
        sparse_params = c.get("sparse_params", {}).copy()

        if "alpha" in c:
            sparse_params["alpha"] = c["alpha"]
        if "beta" in c:
            sparse_params["beta"] = c["beta"]

        server = SparseFLServer(**server_args)
        server.run(**sparse_params)

    elif aggregate_type == "fedavg":
        fedavg_params = c.get("fedavg_params", {}).copy()
        alpha = fedavg_params.get("alpha", c.get("alpha"))

        server = FedAvgServer(**server_args)
        server.run(alpha=alpha)

    else:
        raise ValueError(f"Unknown aggregate_type: {aggregate_type}")


# ==========================================================
# SWEEP AGENT
# ==========================================================

def run_sweep_agent(agent_id, runs, project_name, training_config, total_agents):
    runs_for_agent = [
        run for i, run in enumerate(runs)
        if i % total_agents == agent_id
    ]
    print(f"[Agent {agent_id}] Runs: {len(runs_for_agent)}")

    for run_config in runs_for_agent:
        try:
            combined_config = nest_dot_keys({**training_config, **run_config})

            wandb.init(project=project_name, config=combined_config)

            model_name = combined_config.get("model_name")
            model      = get_model(model_name)

            if model_name == "Llama-3-8B":
                rank  = combined_config.get("rolora_params", {}).get("rank", 16)
                # ✅ FIX: use helper that also enables gradient checkpointing
                model = prepare_llama_model(model, rank=rank)

            train(wandb.config, model)
            wandb.finish()

        except Exception as e:
            print(f"[Agent {agent_id}] Error: {e}")
            wandb.finish(exit_code=1)


# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",             type=str, required=True)
    parser.add_argument("--num-agents",         type=int, default=1)
    parser.add_argument("--num-run-per-config", type=int, default=1)
    args = parser.parse_args()

    print(f"[INFO] Loading config: {args.config}")
    config = load_config_from_yaml(args.config)

    if config is None:
        sys.exit(1)

    # ----------------------------------------------------------
    # SWEEP MODE
    # ----------------------------------------------------------
    if "sweep_config" in config and "training_config" in config:
        print("[MODE] Sweep Mode")

        runs = generate_runs_from_sweep_config(
            config["sweep_config"],
            args.num_run_per_config
        )

        Parallel(n_jobs=args.num_agents)(
            delayed(run_sweep_agent)(
                agent_id=i,
                runs=runs,
                project_name=config["training_config"]["project_name"],
                training_config=config["training_config"],
                total_agents=args.num_agents,
            )
            for i in range(args.num_agents)
        )

    # ----------------------------------------------------------
    # SINGLE RUN MODE
    # ----------------------------------------------------------
    elif "training_config" in config:
        print("[MODE] Single Run")

        t_config   = config["training_config"]
        model_name = t_config.get("model_name")

        wandb.init(
            project=t_config.get("project_name", "federated_project"),
            config=t_config,
        )

        print(f"[ACTION] Loading model: {model_name}")
        model = get_model(model_name)

        if model_name == "Llama-3-8B":
            print("[ACTION] Applying RoLoRA...")
            rank  = t_config.get("rolora_params", {}).get("rank", 16)
            # ✅ FIX: use helper that also enables gradient checkpointing
            model = prepare_llama_model(model, rank=rank)

        train(wandb.config, model)

    else:
        print("[ERROR] Invalid config structure — must contain 'training_config'")
        sys.exit(1)


# ==========================================================
# ENTRY
# ==========================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()