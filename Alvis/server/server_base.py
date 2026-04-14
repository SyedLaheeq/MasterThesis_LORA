import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import wandb

from models.model import DeeperCIFARCNN
from client.client import Client
from attack.attack import Attack
from defence.defence import Defence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseServer:
    """
    Base Server class that handles common functionalities like:
      - Dataset loading
      - Client initialization
      - Dataset distribution
      - Gathering client updates
      - Accuracy calculation
    """

    def __init__(
        self,
        dataset_name,
        num_clients,
        fraction_malicious,
        attack_args=None,
        defence_args=None,
        total_epochs=5,
        q_factor=1.0,
        model=None,
        evaluate_each_epoch=2,
        local_epochs=1,
        batch_size=64,
        malicious_type="group_oriented",
        device="cpu",
        multi_attack_args=None,
        normalize_params=False,
        learning_rate=1e-4,     # ✅ FIXED: explicit lr param, passed down to clients
    ):
        assert malicious_type in ["group_oriented", "random"]

        if model is None:
            model = DeeperCIFARCNN()

        self.device = torch.device(
            "cuda" if device in ["cuda", "gpu"] and torch.cuda.is_available() else "cpu"
        )
        self.global_model = model.to(self.device)

        self.q_factor = q_factor
        self.learning_rate = learning_rate   # ✅ store for use in gather
        print(f"[INFO] q_factor      = {q_factor}")
        print(f"[INFO] learning_rate = {learning_rate}")

        self.num_clients = num_clients
        self.fraction_malicious = fraction_malicious
        self.total_epochs = total_epochs
        self.local_epochs = local_epochs
        self.evaluate_each_epoch = evaluate_each_epoch

        self.defence_args = defence_args or {"defence_type": "no_defence"}
        self.defence_func = Defence(defence_args=self.defence_args)

        self.attack_args = attack_args
        self.attack_func = Attack(attack_args=attack_args) if attack_args else None

        self.multi_attack_args = multi_attack_args
        self.normalize_params = normalize_params

        self.clients = self._initialize_clients(
            dataset_name=dataset_name,
            num_clients=num_clients,
            model=model,
            fraction_malicious=fraction_malicious,
            attack_args=attack_args,
            q_factor=q_factor,
            batch_size=batch_size,
            malicious_type=malicious_type,
            learning_rate=learning_rate,    # ✅ pass through
        )
        print("[DEBUG] Client malicious flags:", [c.malicious for c in self.clients])

        self.list_m_next = []
        self.list_w_next = []

    # ==========================================================
    # CLIENT INITIALIZATION
    # ==========================================================
    def _initialize_clients(
        self, dataset_name, num_clients, model, fraction_malicious,
        attack_args, q_factor, batch_size, malicious_type, learning_rate
    ):
        self.num_clients = num_clients
        self.train_dataset, self.test_dataset = self._load_dataset(dataset_name)

        num_label = max(self.train_dataset.targets.tolist()) + 1
        client_loaders, group2client_idx = self._distribute_dataset(
            self.train_dataset, num_label=num_label, q_factor=q_factor, batch_size=batch_size
        )

        # ----------------------------------------------------------
        # MULTI-ATTACK MODE
        # ----------------------------------------------------------
        if self.multi_attack_args is not None and len(self.multi_attack_args) > 0:
            total_frac = sum(d['fraction_malicious'] for d in self.multi_attack_args)
            if total_frac != 1.0:
                raise ValueError(
                    f"Sum of 'fraction_malicious' in multi_attack_args must equal 1.0 (got {total_frac})."
                )

            client_attack_args = [None] * num_clients
            total_malicious_count = int(fraction_malicious * num_clients)

            if malicious_type == "random":
                malicious_indices = set(random.sample(range(num_clients), total_malicious_count))
            elif malicious_type == "group_oriented":
                num_group_mal = int(fraction_malicious * num_label)
                group_malicious_ids = random.sample(range(num_label), num_group_mal)
                mal_list = []
                for g_id in group_malicious_ids:
                    mal_list.extend(np.where(np.array(group2client_idx) == g_id)[0].tolist())
                if len(mal_list) < total_malicious_count:
                    g_id = random.sample(set(range(num_label)) - set(group_malicious_ids), 1)
                    mal_list.extend(np.where(np.array(group2client_idx) == g_id)[0].tolist())
                if len(mal_list) > total_malicious_count:
                    mal_list = random.sample(mal_list, total_malicious_count)
                malicious_indices = set(mal_list)
            else:
                raise ValueError(f"Unknown malicious_type: {malicious_type}")

            all_malicious = list(malicious_indices)
            random.shuffle(all_malicious)

            split_points = []
            running_sum = 0.0
            for attack_dict in self.multi_attack_args:
                running_sum += attack_dict['fraction_malicious']
                split_points.append(int(running_sum * len(all_malicious)))

            prev = 0
            for attack_dict, next_split in zip(self.multi_attack_args, split_points):
                chosen = all_malicious[prev:next_split]
                print(f"Malicious Client Indices attack {attack_dict['attack_type']}: {chosen}")
                wandb.log({f"malicious_clients_{attack_dict['attack_type']}": chosen})
                for c in chosen:
                    client_attack_args[c] = attack_dict
                prev = next_split

            clients = []
            for i in range(num_clients):
                is_malicious = (client_attack_args[i] is not None)
                clients.append(Client(
                    client_id=i,
                    model=model,
                    train_dataset=client_loaders[i].dataset,  # ✅ FIXED: pass dataset not loader
                    malicious=is_malicious,
                    attack_args=client_attack_args[i],
                    local_epochs=self.local_epochs,            # ✅ FIXED: correct param name
                    learning_rate=learning_rate,               # ✅ FIXED: pass lr
                    batch_size=batch_size,
                ))
            return clients

        # ----------------------------------------------------------
        # SINGLE ATTACK MODE
        # ----------------------------------------------------------
        num_malicious = int(fraction_malicious * num_clients)

        if malicious_type == "random":
            malicious_ids = random.sample(range(num_clients), num_malicious)
        elif malicious_type == "group_oriented":
            num_group_malicious = int(fraction_malicious * num_label)
            group_malicious_ids = random.sample(range(num_label), num_group_malicious)
            malicious_ids = []
            for group_malicious_id in group_malicious_ids:
                malicious_ids.extend(
                    np.where(np.array(group2client_idx) == group_malicious_id)[0].tolist()
                )
            if len(malicious_ids) < num_malicious:
                group_malicious_id = random.sample(
                    sorted(set(range(num_label)) - set(group_malicious_ids)), 1
                )
                ids = np.where(np.array(group2client_idx) == group_malicious_id)[0].tolist()
                malicious_ids.extend(random.sample(ids, num_malicious - len(malicious_ids)))
            if len(malicious_ids) > num_malicious:
                malicious_ids = random.sample(malicious_ids, num_malicious)

        print(f"Malicious Client Indices: {malicious_ids}")
        wandb.log({"malicious_clients": malicious_ids})

        clients = []
        for i in range(num_clients):
            is_malicious = (i in malicious_ids)
            clients.append(Client(
                client_id=i,
                model=model,
                train_dataset=client_loaders[i].dataset,  # ✅ FIXED: pass dataset not loader
                malicious=is_malicious,
                attack_args=attack_args,
                local_epochs=self.local_epochs,            # ✅ FIXED: correct param name
                learning_rate=learning_rate,               # ✅ FIXED: pass lr
                batch_size=batch_size,
            ))
        return clients

    # ==========================================================
    # DATASET LOADING
    # ==========================================================
    def _load_dataset(self, dataset_name):
        if dataset_name == "MNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)

        elif dataset_name == "EMNIST":
            transform = transforms.Compose([transforms.ToTensor()])
            split = 'balanced'
            train_dataset = datasets.EMNIST('./data', train=True, split=split, download=True, transform=transform)
            test_dataset  = datasets.EMNIST('./data', train=False, split=split, download=True, transform=transform)

        elif dataset_name == "CIFAR10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
            test_dataset  = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
            train_dataset.targets = torch.tensor(train_dataset.targets)
            test_dataset.targets  = torch.tensor(test_dataset.targets)

        elif dataset_name == "alpaca":
            from datasets import load_dataset
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
            tokenizer.pad_token = tokenizer.eos_token

            raw_dataset = load_dataset("tatsu-lab/alpaca", split='train')

            def tokenize_function(examples):
                texts = []
                for i, inp, out in zip(examples['instruction'], examples['input'], examples['output']):
                    if inp.strip():
                        texts.append(f"### Instruction:\n{i}\n\n### Input:\n{inp}\n\n### Response:\n{out}")
                    else:
                        texts.append(f"### Instruction:\n{i}\n\n### Response:\n{out}")
                return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

            print("[INFO] Tokenizing Alpaca dataset...")
            tokenized_dataset = raw_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=raw_dataset.column_names
            )
            tokenized_dataset.set_format("torch")

            split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split['train'].select(range(5000))
            test_dataset  = split['test'].select(range(200))

            def build_semantic_targets(dataset, num_groups):
                targets = []
                for example in dataset:
                    input_ids = torch.tensor(example["input_ids"])
                    length = (input_ids != 0).sum().item()
                    if length < 40:
                        group = 0
                    elif length < 80:
                        group = 1
                    elif length < 120:
                        group = 2
                    else:
                        group = 3
                    targets.append(group)
                return torch.tensor(targets)

            num_groups = min(4, self.num_clients)
            train_dataset.targets = build_semantic_targets(train_dataset, num_groups)
            test_dataset.targets  = build_semantic_targets(test_dataset, num_groups)

            print(f"[SUCCESS] Loaded Alpaca: {len(train_dataset)} train, {len(test_dataset)} test")
            return train_dataset, test_dataset

        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        return train_dataset, test_dataset

    # ==========================================================
    # DATASET DISTRIBUTION
    # ==========================================================
    def _distribute_dataset(self, train_dataset, num_label, q_factor, batch_size):
        num_label = min(num_label, self.num_clients)

        if q_factor >= 0.99:
            indices = np.random.permutation(len(train_dataset))
            splits  = np.array_split(indices, self.num_clients)
            client_loaders = []
            for i in range(self.num_clients):
                subset = torch.utils.data.Subset(train_dataset, splits[i])
                loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
                client_loaders.append(loader)
            print("[INFO] Using IID data distribution")
            return client_loaders, [0] * self.num_clients

        targets    = np.array(train_dataset.targets)
        label2idx  = {i: np.where(targets == i)[0] for i in range(num_label)}

        group2client_idx = []
        for i in range(num_label):
            group2client_idx.extend([i] * (self.num_clients // num_label))

        remainder = self.num_clients - len(group2client_idx)
        if remainder > 0:
            extra_groups = list(np.random.permutation(num_label)[:remainder])
            group2client_idx.extend(extra_groups)

        random.shuffle(group2client_idx)

        group2data_idx = [[] for _ in range(num_label)]
        for i in range(num_label):
            perm_idx        = np.random.permutation(len(label2idx[i]))
            split_point     = int(len(perm_idx) * q_factor)
            q_group_idx     = perm_idx[:split_point]
            q_complement_idx = perm_idx[split_point:]

            group2data_idx[i].extend(label2idx[i][q_group_idx.tolist()])

            if len(q_complement_idx) > 0:
                splitted = np.array_split(q_complement_idx, num_label - 1)
                c_idx = 0
                for j in range(num_label):
                    if j != i:
                        group2data_idx[j].extend(label2idx[i][splitted[c_idx].tolist()])
                        c_idx += 1

        for sublist in group2data_idx:
            random.shuffle(sublist)

        client2data_idx = {}
        for i in range(num_label):
            client_indices = np.where(np.array(group2client_idx) == i)[0]
            data_indices   = group2data_idx[i]
            splitted_data  = np.array_split(data_indices, len(client_indices))
            for j, c in enumerate(client_indices):
                client2data_idx[c] = list(splitted_data[j])

        min_size = min(len(v) for v in client2data_idx.values())
        for k in client2data_idx:
            client2data_idx[k] = client2data_idx[k][:min_size]

        client_loaders = []
        for i in range(self.num_clients):
            subset_ds = torch.utils.data.Subset(train_dataset, client2data_idx[i])
            loader    = torch.utils.data.DataLoader(subset_ds, batch_size=batch_size, shuffle=True)
            client_loaders.append(loader)

        print(f"[INFO] Using non-IID distribution | q_factor={q_factor}")
        return client_loaders, group2client_idx

    # ==========================================================
    # GATHER CLIENT UPDATES
    # ==========================================================
    def _gather_client_updates(
        self,
        global_weights,
        epoch,
        lr,
        return_avg_loss=True,
        compute_gradient=True,
        return_params=False,
        apply_attack=True,
    ):
        client_gradients = []
        client_losses    = []

        # ----------------------------------------------------------
        # STEP 1: COLLECT ALL CLIENT UPDATES
        # ----------------------------------------------------------

        # ✅ CRITICAL FIX: state_dict() returns tensors that SHARE storage
        # with model parameters. When optimizer.step() updates param.data it
        # simultaneously mutates global_weights — making every delta = 0.
        # Clone only LoRA params (the only ones used in delta computation).
        frozen_global_weights = {
            k: v.detach().clone()
            for k, v in global_weights.items()
            if "lora_" in k
        }

        for client in self.clients:
            print(f"[Server] Processing Client {client.client_id}", flush=True)

            updates, avg_loss = client.local_update(
                global_weights=frozen_global_weights,
                epoch=epoch,
                return_avg_loss=return_avg_loss,
                compute_gradient=compute_gradient,
                return_params=return_params,
                lr=self.learning_rate,   # ✅ FIXED: always use stored lr, not alpha
                server_device=self.device,
            )

            client_gradients.append(updates)
            client_losses.append(avg_loss)

        # ----------------------------------------------------------
        # STEP 2: REMOVE NaN / INF / EMPTY CLIENTS
        # ----------------------------------------------------------
        clean_gradients = []
        clean_losses    = []
        clean_ids       = []
        clean_clients   = []

        for i, (g, loss) in enumerate(zip(client_gradients, client_losses)):
            client = self.clients[i]

            if g is None:
                print(f"[Defence] Dropping Client {client.client_id} (None update)")
                continue

            if loss is None or np.isnan(loss) or np.isinf(loss):
                print(f"[Defence] Dropping Client {client.client_id} (NaN/Inf loss)")
                continue

            g_flat = self._flatten_single_gradient(g)
            if g_flat is None:
                print(f"[Defence] Dropping Client {client.client_id} (empty gradient)")
                continue

            if torch.isnan(g_flat).any() or torch.isinf(g_flat).any():
                print(f"[Defence] Dropping Client {client.client_id} (NaN/Inf gradient)")
                continue

            clean_gradients.append(g)
            clean_losses.append(loss)
            clean_ids.append(client.client_id)
            clean_clients.append(client)

        if len(clean_gradients) == 0:
            raise RuntimeError("All clients dropped due to NaNs/Infs!")

        # ----------------------------------------------------------
        # STEP 3: APPLY ATTACK (ONLY ON CLEAN CLIENTS)
        # ----------------------------------------------------------
        if self.attack_func is not None and apply_attack:
            attack_type = self.attack_args.get("attack_type", "none")

            if attack_type == "lie_attack":
                # lie_attack returns (grads, losses) — it also crafts malicious losses
                result = self.attack_func(
                    clean_gradients,
                    clean_clients,
                    losses=clean_losses,
                    z=self.attack_args.get("z", None),
                )
                if isinstance(result, tuple):
                    client_gradients, client_losses = result
                else:
                    client_gradients = result
                    client_losses    = clean_losses
            else:
                # all other attacks only modify gradients
                client_gradients = self.attack_func(
                    clean_gradients,
                    clean_clients,
                    scale=self.attack_args.get("scale", 20.0),
                )
                client_losses = clean_losses
        else:
            client_gradients = clean_gradients
            client_losses    = clean_losses

        # ----------------------------------------------------------
        # DEBUG: NORM PER CLIENT
        # ----------------------------------------------------------
        print("---- ATTACK CHECK ----")
        for g, client in zip(client_gradients, clean_clients):
            total_norm = torch.sqrt(
                sum(torch.norm(v.to(torch.float32)) ** 2 for v in g.values())
            ).item()
            print(f"Client {client.client_id} | malicious={client.malicious} | norm={total_norm:.6f}")

        return client_gradients, client_losses, clean_ids

    # ==========================================================
    # ACCURACY / LOSS EVALUATION
    # ==========================================================
    def calculate_accuracy(self, is_fedavg=False):
        """
        For LLM (Alpaca): computes average cross-entropy loss on the test set.
        For image models: computes classification accuracy.
        """
        self.global_model.eval()

        # ----------------------------------------------------------
        # LLM path (Alpaca / causal LM)
        # ----------------------------------------------------------
        is_llm = not hasattr(self.global_model, 'conv1')   # rough check

        if is_llm:
            loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size if hasattr(self, 'batch_size') else 8,
                shuffle=False,
            )
            total_loss   = 0.0
            total_batches = 0

            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    outputs = self.global_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,   # ✅ consistent with training
                    )

                    if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                        continue

                    total_loss    += outputs.loss.item()
                    total_batches += 1

            if total_batches == 0:
                print("[EVAL] All batches were NaN — model may have diverged")
                return 0.0, float('inf')

            avg_loss = total_loss / total_batches
            print(f"[EVAL] Test Loss: {avg_loss:.4f}")
            return 0.0, avg_loss

        # ----------------------------------------------------------
        # Image classification path (MNIST / CIFAR10)
        # ----------------------------------------------------------
        loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=128, shuffle=False)
        total_correct = 0
        total_samples = 0
        total_loss    = 0.0
        loss_fn       = nn.CrossEntropyLoss()

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                out  = self.global_model(X)
                loss = loss_fn(out, y)

                _, preds = torch.max(out, dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)
                total_loss    += loss.item()

        accuracy = 100.0 * total_correct / total_samples
        avg_loss = total_loss / len(loader)

        if np.isnan(avg_loss) or avg_loss > 9:
            raise RuntimeError(f"Model diverged: Loss = {avg_loss}")

        print(f"[EVAL] Test Accuracy = {accuracy:.2f}%, Test Loss = {avg_loss:.4f}")
        return accuracy, avg_loss

    # ==========================================================
    # HELPERS
    # ==========================================================
    def _flatten_single_gradient(self, grad_dict):
        if grad_dict is None or len(grad_dict) == 0:
            return None
        flat = []
        for k in sorted(grad_dict.keys()):
            g = grad_dict[k]
            if g is None:
                continue
            flat.append(g.view(-1))
        if len(flat) == 0:
            return None
        return torch.cat(flat)

    def _flatten_tensors(self, input_list):
        flattened = []
        for tensors in input_list:
            if isinstance(tensors, dict):
                sorted_keys = sorted(tensors.keys())
                cat = torch.cat([tensors[k].view(-1) for k in sorted_keys])
            else:
                cat = torch.cat([t.view(-1) for t in tensors])
            flattened.append(cat)
        return torch.stack(flattened).T

    def _normalize_gradients(self, client_gradients):
        grad_norms = [
            torch.norm(torch.cat([g.view(-1) for g in grad])).item()
            for grad in client_gradients
        ]
        median_norm = np.median(grad_norms)
        for i, grad in enumerate(client_gradients):
            if grad_norms[i] > 0:
                scale_factor = median_norm / grad_norms[i]
                for j in range(len(grad)):
                    grad[j] = grad[j] * scale_factor

    def _split(self, arr, n):
        k, m = divmod(len(arr), n)
        return (arr[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n))