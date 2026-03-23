# import os
# os.environ["WANDB_START_METHOD"] = "thread"
# # Ensure CUDA_VISIBLE_DEVICES is set and process it
# GPUS = os.environ.get("CUDA_VISIBLE_DEVICES", "")
# if GPUS:
#     GPUS = [int(gpu) for gpu in GPUS.split(",")]
# else:
#     GPUS = []  # Default to an empty list if CUDA_VISIBLE_DEVICES is not set

# import sys
# import argparse
# import wandb
# import yaml
# import multiprocessing as mp
# from joblib import Parallel, delayed
# from itertools import product


# def load_config_from_yaml(filepath):
#     """Load configuration from YAML."""
#     with open(filepath, 'r') as file:
#         return yaml.safe_load(file)

# def write_config_to_yaml(filepath, config):
#     """Write configuration to YAML."""
#     with open(filepath, 'w') as file:
#         yaml.safe_dump(config, file, sort_keys=False)

# def get_model(model_name):
#     """Retrieve model instance by name."""
#     from model.model import DeeperCIFARCNN, ThreeLayerFC
#     # Mapping of model names to classes
#     MODEL_MAP = {
#         "DeeperCIFARCNN": DeeperCIFARCNN,
#         "ThreeLayerFC": ThreeLayerFC,
#     }

#     if model_name in MODEL_MAP:
#         return MODEL_MAP[model_name]()
#     else:
#         raise ValueError(f"Unknown model name: {model_name}")

# def train(config, model):
#     """
#     Train model based on configuration.

#     Automatically retains default sparse_params (alpha, beta, lambda_max)
#     if not overridden in the config (e.g., by a W&B sweep).
#     """
#     from server.server_sparse import SparseFLServer
#     from server.server_fedavg import FedAvgServer

#     dataset_name = config.get("dataset_name", "MNIST")
#     num_clients = config.get("num_clients", 10)
#     fraction_malicious = config.get("fraction_malicious", 0.0)
#     total_epochs = config.get("total_epochs", 10)
#     q_factor = config.get("q_factor", 0.1)
#     evaluate_each_epoch = config.get("evaluate_each_epoch", 1)
#     attack_args = config.get("attack_args", None)
#     defence_args = config.get("defence_args", None)
#     aggregate_type = config.get("aggregate_type", "fedavg")
#     batch_size = config.get("batch_size", 64)
#     local_epochs = config.get("local_epochs", 1)
#     malicious_type = config.get("malicious_type", "group_oriented")
#     device = config.get("device", "cpu")
#     multi_attack_args = config.get("multi_attack_args", None)
#     normalize_params = config.get("normalize_params", False)

#     # Common server arguments
#     server_args = {
#         "dataset_name": dataset_name,
#         "num_clients": num_clients,
#         "fraction_malicious": fraction_malicious,
#         "attack_args": attack_args,
#         "defence_args": defence_args,
#         "total_epochs": total_epochs,
#         "q_factor": q_factor,
#         "model": model,
#         "evaluate_each_epoch": evaluate_each_epoch,
#         "batch_size": batch_size,
#         "local_epochs": local_epochs,
#         "malicious_type": malicious_type,
#         "device": device,
#         "multi_attack_args": multi_attack_args,
#         "normalize_params": normalize_params,
#     }

#     if aggregate_type == "sparse":
#         # Copy the dictionary so we don't mutate the original
#         sparse_params = config.get("sparse_params", {}).copy()

#         # Conditionally override alpha, beta, lambda_max if present in config
#         if "alpha" in config:
#             sparse_params["alpha"] = config["alpha"]
#         if "beta" in config:
#             sparse_params["beta"] = config["beta"]
#         if "lambda_max" in config:
#             # Keep the same structure but replace only the middle value
#             old_lambda = sparse_params.get("lambda_val", [0, 0.0025, 100])
#             sparse_params["lambda_val"] = (old_lambda[0], config["lambda_max"], old_lambda[2])

#         server = SparseFLServer(**server_args)
#         server.run(**sparse_params)

#     elif aggregate_type == "fedavg":
#         fedavg_params = config.get("fedavg_params", {}).copy()

#         # Make sure alpha is present
#         if "alpha" in config:
#             fedavg_params["alpha"] = config["alpha"]
#         alpha = fedavg_params.get("alpha")
#         if alpha is None:
#             raise ValueError("FedAvgServer.run() requires an 'alpha' parameter.")
        
#         server = FedAvgServer(**server_args)
#         server.run(alpha=alpha)

#     else:
#         raise ValueError(f"Unknown aggregate_type: {aggregate_type}")

# def nest_dot_keys(config_dict):
#     """
#     Convert keys containing dots (e.g. "sparse_params.alpha") 
#     into nested dictionaries {"sparse_params": {"alpha": ...}} recursively.
#     """
#     def insert_nested(d, keys, value):
#         """
#         Recursively insert a value into the nested dictionary `d` based on `keys`.
#         """
#         if len(keys) == 1:
#             d[keys[0]] = value
#         else:
#             if keys[0] not in d:
#                 d[keys[0]] = {}
#             insert_nested(d[keys[0]], keys[1:], value)

#     nested = {}
#     for key, value in config_dict.items():
#         parts = key.split(".")
#         insert_nested(nested, parts, value)

#     return nested

# def generate_runs_from_sweep_config(sweep_config, num_run_per_config):
#     """
#     Generate all possible runs based on the sweep configuration.
#     Only supports grid sweeps for simplicity.
#     """
#     params = sweep_config.get("parameters", {})
#     param_names = list(params.keys())
#     param_values = [param["values"] for param in params.values()]

#     # Generate cartesian product of all parameter values
#     param_combinations = list(product(*param_values))

#     # Create a list of dicts for each combination
#     runs = [
#         {param_name: value for param_name, value in zip(param_names, combination)}
#         for combination in param_combinations
#     ] * num_run_per_config

#     return runs

# def run_sweep_agent_manual(agent_id, runs, project_name, training_config, total_agents, num_gpus, ignore_default_params):
#     """
#     Manually execute a subset of runs assigned to this agent.
#     """
#     global GPUS

#     # Assign GPU for this agent
#     gpu_id = agent_id % num_gpus if num_gpus > 0 else None
#     if gpu_id is not None:
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUS[gpu_id])
#         print(f"[Agent {agent_id}] Assigned to GPU {str(GPUS[gpu_id])}")
#     else:
#         print(f"[Agent {agent_id}] Running on CPU.")

#     import torch

#     if torch.cuda.is_available():
#         torch.cuda.set_per_process_memory_fraction(0.95, device=0)
#     else:
#         print(f"[Agent {agent_id}] Running on CPU.")

#     # Assign runs to this agent
#     runs_for_agent = [run for idx, run in enumerate(runs) if idx % total_agents == agent_id]
#     print(f"[Agent {agent_id}] Assigned runs: {len(runs_for_agent)}")

#     for run_id, run_config in enumerate(runs_for_agent):
#         try:
#             # Merge the run configuration with the base training configuration
#             combined_config = nest_dot_keys({**training_config, **run_config})

#             num_clients = combined_config.get("num_clients", None)
#             fraction_malicious = combined_config.get("fraction_malicious", None)
#             defence_args = combined_config.get("defence_args", None)

#             if defence_args:
#                 f = int(fraction_malicious * num_clients)

#                 # Adjust Krum factor
#                 if defence_args['defence_type'] in ['krum', 'bulyan'] and not ignore_default_params:
#                     combined_config['defence_args']['krum_factor'] = num_clients - f - 2

#                 # Adjust Trimmed Mean factor
#                 if defence_args['defence_type'] in ['trimmed_mean'] and not ignore_default_params:
#                     combined_config['defence_args']['trimmed_factor'] = fraction_malicious

#                 # Adjust Huber delta
#                 if defence_args['defence_type'] == 'huber' and not ignore_default_params:
#                     # ε  – upper bound on Byzantine fraction
#                     combined_config['defence_args']['byzantine_frac'] = fraction_malicious

#             # Initialize WandB manually with the fetched config
#             wandb.init(
#                 project=project_name,
#                 config=combined_config,
#                 settings=wandb.Settings(start_method="thread"),
#             )

#             # Load model and train
#             model = get_model(combined_config.get("model_name", "DeeperCIFARCNN"))
#             train(combined_config, model)

#             wandb.finish()

#         except Exception as e:
#             print(f"[Agent {agent_id}] Error occurred: {e}")
#             wandb.finish(exit_code=1)
#             continue

# def main():
#     global GPUS

#     parser = argparse.ArgumentParser(description="Federated Learning Training and Sweeps")
#     parser.add_argument("--config", type=str, help="Path to the .yaml configuration file")
#     parser.add_argument("--write-config", type=str, help="Path to save the default configuration as .yaml")
#     parser.add_argument("--num-agents", type=int, default=1, help="Number of parallel agents to run")
#     parser.add_argument("--num-gpus", type=int, default=0, help="Number of gpus to run")
#     parser.add_argument("--num-run-per-config", type=int, default=1, help="Number of runs per each config")
#     parser.add_argument("--ignore-default-params", action="store_true", help="Ignore default params for krum and trimmed mean")


#     args = parser.parse_args()

#     # Default configuration
#     default_config = {
#         "training_config": {
#             "project_name": "federated_learning_project",
#             "model_name": "DeeperCIFARCNN",
#             "aggregate_type": "sparse",
#             "dataset_name": "CIFAR10",
#             "num_clients": 200,
#             "fraction_malicious": 0.4,
#             "total_epochs": 200,
#             "q_factor": 0.8,
#             "evaluate_each_epoch": 1,
#             "attack_args": {"attack_type": "flip_labels", "attack_epoch": 5, "max_label": 9},
#             "defence_args": {"defence_type": "no_defence"},
#             "batch_size": 64,
#             "local_epochs": 3,
#             "malicious_type": "group_oriented",
#             "sparse_params": {
#                 "alpha": 0.01,
#                 "beta": 1e-4,
#                 "is_ftotal": True,
#                 "lambda_val": [0, 0.0025, 100],
#                 "c_alpha": 1e-3,
#                 "rho_alpha": 0.5,
#                 "max_line_search_iterations_alpha": 0,
#                 "c_beta": 1e-3,
#                 "rho_beta": 0.5,
#                 "max_line_search_iterations_beta": 0,
#             },
#             "fedavg_params": {"alpha": 0.01},
#         },
#         "sweep_config": {
#             "method": "bayes",
#             "metric": {"name": "test_accuracy", "goal": "maximize"},
#             "parameters": {
#                 "sparse_params.alpha": {"values": [0.025, 0.01, 0.006, 0.0015]},
#                 "sparse_params.beta": {"values": [8e-4, 6e-4, 4e-4, 2e-4]},
#                 "sparse_params.lambda_val": {"values": [(0, 0.004, 100), (0, 0.003, 100)]},
#             },
#         },
#     }

#     # Handle --write-config
#     if args.write_config:
#         write_config_to_yaml(args.write_config, default_config)
#         print(f"Default configuration written to {args.write_config}")
#         sys.exit()

#     # Ensure we have a config file unless we wrote one
#     if not args.config:
#         print("Error: --config must be provided unless --write-config is used.")
#         sys.exit(1)

#     config = load_config_from_yaml(args.config)

#     # We will spawn multiple processes if --num-agents > 1
#     num_agents = args.num_agents
#     num_gpus = args.num_gpus
#     num_run_per_config = args.num_run_per_config
#     ignore_default_params = args.ignore_default_params

#     processes = []
#     if len(GPUS) < num_gpus:
#         GPUS = list(range(num_gpus))
    
#     assert len(GPUS) == num_gpus

#     # Check for both sweep_config and training_config
#     if "sweep_config" in config and "training_config" in config:
#         sweep_config = config["sweep_config"]
#         training_config = config["training_config"]
#         project_name = training_config["project_name"]

#         # Generate all runs from the sweep configuration
#         runs = generate_runs_from_sweep_config(sweep_config, num_run_per_config)
#         print(f"Total runs generated: {len(runs)}")

#         # Use Joblib to parallelize agents
#         Parallel(n_jobs=num_agents)(
#             delayed(run_sweep_agent_manual)(
#                 agent_id=agent_id,
#                 runs=runs,
#                 project_name=project_name,
#                 training_config=training_config,
#                 total_agents=num_agents,
#                 num_gpus=num_gpus,
#                 ignore_default_params=ignore_default_params,
#             )
#             for agent_id in range(num_agents)
#         )

#     elif "training_config" in config:
#         training_config = config["training_config"]
#         project_name = training_config["project_name"]

#         wandb.init(project=project_name, config=training_config)
#         model = get_model(training_config["model_name"])
#         train(wandb.config, model)

#     else:
#         print("Invalid configuration file. Must contain both 'training_config' and 'sweep_config' for sweep mode.")


# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     main()
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

# ✅ Correct imports
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
    params = sweep_config.get("parameters", {})
    param_names = list(params.keys())
    param_values = [param["values"] for param in params.values()]

    combinations = list(product(*param_values))

    runs = [
        {k: v for k, v in zip(param_names, combo)}
        for combo in combinations
    ]

    return runs * num_run_per_config


# ==========================================================
# TRAIN FUNCTION
# ==========================================================

def train(config, model):
    from server.server_sparse import SparseFLServer
    from server.server_fedavg import FedAvgServer

    c = config if hasattr(config, "get") else config.as_dict()

    dataset_name = c.get("dataset_name", "MNIST")
    num_clients = c.get("num_clients", 10)
    fraction_malicious = c.get("fraction_malicious", 0.0)
    total_epochs = c.get("total_epochs", 10)
    q_factor = c.get("q_factor", 0.1)
    evaluate_each_epoch = c.get("evaluate_each_epoch", 1)
    attack_args = c.get("attack_args", None)
    defence_args = c.get("defence_args", None)
    aggregate_type = c.get("aggregate_type", "fedavg")
    batch_size = c.get("batch_size", 64)
    local_epochs = c.get("local_epochs", 1)
    malicious_type = c.get("malicious_type", "group_oriented")
    normalize_params = c.get("normalize_params", False)

    # 🔥 FORCE SAFE DEVICE
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[DEBUG] Device: {device}")
    print(f"[DEBUG] Attack: {attack_args}")

    server_args = {
        "dataset_name": dataset_name,
        "num_clients": num_clients,
        "fraction_malicious": fraction_malicious,
        "attack_args": attack_args,
        "defence_args": defence_args,
        "total_epochs": total_epochs,
        "q_factor": q_factor,
        "model": model,
        "evaluate_each_epoch": evaluate_each_epoch,
        "batch_size": batch_size,
        "local_epochs": local_epochs,
        "malicious_type": malicious_type,
        "device": device,
        "normalize_params": normalize_params,
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

            wandb.init(
                project=project_name,
                config=combined_config
            )

            model_name = combined_config.get("model_name")
            model = get_model(model_name)

            # 🔥 Apply RoLoRA for LLaMA
            if model_name == "Llama-3-8B":
                rank = combined_config.get("rolora_params", {}).get("rank", 16)
                model = inject_rolora_to_llama(model, rank=rank)

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
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--num-run-per-config", type=int, default=1)

    args = parser.parse_args()

    print(f"[INFO] Loading config: {args.config}")
    config = load_config_from_yaml(args.config)

    if config is None:
        sys.exit(1)

    # =========================
    # SWEEP MODE
    # =========================
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
                total_agents=args.num_agents
            )
            for i in range(args.num_agents)
        )

    # =========================
    # SINGLE RUN MODE
    # =========================
    elif "training_config" in config:
        print("[MODE] Single Run")

        t_config = config["training_config"]
        model_name = t_config.get("model_name")

        wandb.init(
            project=t_config.get("project_name", "federated_project"),
            config=t_config
        )

        print(f"[ACTION] Loading model: {model_name}")
        model = get_model(model_name)

        # 🔥 RoLoRA
        if model_name == "Llama-3-8B":
            print("[ACTION] Applying RoLoRA...")
            rank = t_config.get("rolora_params", {}).get("rank", 16)
            model = inject_rolora_to_llama(model, rank=rank)

        train(wandb.config, model)

    else:
        print("[ERROR] Invalid config structure")
        sys.exit(1)


# ==========================================================
# ENTRY
# ==========================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()