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
    It does NOT implement a specific aggregation strategy.
    """

    ATTACK_ON_BENIGN_UPDATES = ['lie_attack']

    def __init__(
        self, 
        dataset_name,
        num_clients,
        fraction_malicious,
        attack_args=None,
        defence_args=None,
        total_epochs=5,
        q_factor=0.1,
        model=DeeperCIFARCNN(),
        evaluate_each_epoch=2,
        local_epochs=1,
        batch_size=64,
        malicious_type="group_oriented",
        device="cpu",
        multi_attack_args=None,
        normalize_params=False,
    ):
        """
        Parameters
        ----------
        dataset_name : str
            Name of the dataset ('MNIST', 'CIFAR10', etc.).
        num_clients : int
            Number of clients.
        fraction_malicious : float
            Fraction of clients that are malicious (0 to 1).
        attack_args : dict, optional
            Attack configuration.
        defence_args : dict, optional
            Defence configuration.
        total_epochs : int
            Number of global training rounds (epochs).
        q_factor : float
            Parameter for partial data sharing across classes.
        model : nn.Module, optional
            Model architecture.
        evaluate_each_epoch : int
            Frequency of evaluation on the test set.
        local_epochs : int
            Number of local epochs for each client.
        """
        # Assertion
        assert malicious_type in ["group_oriented", "random"]

        # Models
        # Set device dynamically
        if (device == "gpu" or device == "cuda") and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("WARNING: Using CPU for computation!")
        self.global_model = model.to(self.device)
        self.num_clients = 0
        self.test_dataset = None
        self.fraction_malicious = fraction_malicious

        # Training configuration
        self.total_epochs = total_epochs
        self.local_epochs = local_epochs
        self.evaluate_each_epoch = evaluate_each_epoch
        
        # Defense and Attack
        self.defence_args = defence_args
        if self.defence_args is None:
            self.defence_args = {"defence_type": "no_defence"}
        self.defence_func = Defence(defence_args=self.defence_args)

        self.attack_args = attack_args
        self.attack_func = None
        if attack_args is not None:
            self.attack_type = attack_args.get('attack_type', None)
            self.attack_epoch = attack_args.get('attack_epoch', -1)
            self.attack_func = Attack(attack_args=attack_args)

        self.multi_attack_args = multi_attack_args
        self.normalize_params = normalize_params

        # Initialize clients
        self.clients = self._initialize_clients(
            dataset_name=dataset_name,
            num_clients=num_clients,
            model=model,
            fraction_malicious=fraction_malicious,
            attack_args=attack_args,
            q_factor=q_factor,
            batch_size=batch_size,
            malicious_type=malicious_type
        )

        # For Sparse FL line-search
        self.list_m_next = []
        self.list_w_next = []

    def _initialize_clients(self, dataset_name, num_clients, model, fraction_malicious, attack_args, q_factor, batch_size, malicious_type):
        """
        Loads data, creates client data loaders, and marks some clients as malicious.
        """
        self.num_clients = num_clients
        self.train_dataset, self.test_dataset = self._load_dataset(dataset_name)

        # Number of classes = max label + 1
        num_label = max(self.train_dataset.targets.tolist()) + 1
        client_loaders, group2client_idx = self._distribute_dataset(self.train_dataset, num_label=num_label, q_factor=q_factor, batch_size=batch_size)

        if self.multi_attack_args is not None and len(self.multi_attack_args) > 0:
            # 1) Check that sum of fractions <= 1.0
            total_frac = sum(d['fraction_malicious'] for d in self.multi_attack_args)
            if total_frac != 1.0:
                raise ValueError(
                    "Sum of 'fraction_malicious' in multi_attack_args "
                    f"must equal 1.0 (got {total_frac}). Each attack's fraction "
                    "is relative to the total fraction of malicious clients."
                )

            # 2) Prepare an array of client-specific attack args
            #    (None means benign by default).
            client_attack_args = [None] * num_clients

            # 3) Randomly assign each attack to the appropriate fraction of clients
            total_malicious_count = int(fraction_malicious * num_clients)
            if malicious_type == "random":
                all_indices = list(range(num_clients))
                malicious_indices = set(random.sample(all_indices, total_malicious_count))
            elif malicious_type == "group_oriented":
                num_group_mal = int(fraction_malicious * num_label)
                group_malicious_ids = random.sample(range(num_label), num_group_mal)
                # gather all clients in those groups
                mal_list = []
                for g_id in group_malicious_ids:
                    mal_list.extend(np.where(np.array(group2client_idx) == g_id)[0].tolist())
                if len(mal_list) < total_malicious_count:
                    g_id = random.sample(set(range(num_label)) - set(group_malicious_ids), 1)
                    mal_list.extend(np.where(np.array(group2client_idx) == g_id)[0].tolist())
                # In case we have more clients than total_malicious_count, randomly pick
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

            # 4) Create the Client objects
            clients = []
            for i in range(num_clients):
                is_malicious = (client_attack_args[i] is not None)
                this_attack_args = client_attack_args[i]
                clients.append(Client(
                    client_id=i,
                    model=model,
                    data_loader=client_loaders[i],
                    malicious=is_malicious,
                    attack_args=this_attack_args,
                    local_epoch=self.local_epochs
                ))
            return clients
        else:
            num_malicious = int(fraction_malicious * num_clients)
            if malicious_type == "random":
                malicious_ids = random.sample(range(num_clients), num_malicious)
            elif malicious_type == "group_oriented":
                num_group_malicious = int(fraction_malicious * num_label)
                group_malicious_ids = random.sample(range(num_label), num_group_malicious)
                malicious_ids = []
                for group_malicious_id in group_malicious_ids:
                    malicious_ids.extend(np.where(np.array(group2client_idx) == group_malicious_id)[0].tolist())
                if len(malicious_ids) < num_malicious:
                    group_malicious_id = random.sample(sorted(set(range(num_label)) - set(group_malicious_ids)), 1)
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
                    data_loader=client_loaders[i],
                    malicious=is_malicious,
                    attack_args=attack_args,
                    local_epoch=self.local_epochs
                ))
            return clients

    def _load_dataset(self, dataset_name):
        """
        Loads the specified dataset.
        """
        if dataset_name == "MNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST(
                './data', train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(
                './data', train=False, download=True, transform=transform
            )
        elif dataset_name == "EMNIST":
            transform = transforms.Compose([transforms.ToTensor()])
            split = 'balanced'
            train_dataset = datasets.EMNIST(
                './data', train=True, split=split, 
                download=True, transform=transform
            )
            test_dataset = datasets.EMNIST(
                './data', train=False, split=split, 
                download=True, transform=transform
            )
        elif dataset_name == "CIFAR10": 
            # Common transformations for CIFAR-10 
            transform = transforms.Compose([ 
                # Data augmentation (optional) 
                transforms.ToTensor(), 
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                    (0.2470, 0.2435, 0.2616)) 
            ])
            train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform) 
            test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

            # Convert targets (labels) to a tensor
            train_dataset.targets = torch.tensor(train_dataset.targets)
            test_dataset.targets = torch.tensor(test_dataset.targets)
        elif dataset_name == "alpaca":
            from datasets import load_dataset
            from transformers import AutoTokenizer

            # 1. Load the Tokenizer
            # Use the same model name as in your main script
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
            tokenizer.pad_token = tokenizer.eos_token

            # 2. Load the Raw Alpaca Dataset
            raw_dataset = load_dataset("tatsu-lab/alpaca", split='train')
            

            # 3. Define Tokenization & Prompting
            def tokenize_function(examples):
                # Standard Alpaca Prompt Template
                texts = []
                for i, inp, out in zip(examples['instruction'], examples['input'], examples['output']):
                    if inp.strip():
                        texts.append(f"### Instruction:\n{i}\n\n### Input:\n{inp}\n\n### Response:\n{out}")
                    else:
                        texts.append(f"### Instruction:\n{i}\n\n### Response:\n{out}")
                
                return tokenizer(
                    texts, 
                    truncation=True, 
                    padding="max_length", 
                    max_length=128 # CHANGE TO 512 FRO NORNAL RUM
                )

            # 4. Map the dataset and set to PyTorch format
            print("[INFO] Tokenizing Alpaca dataset... (this may take a minute)")
            tokenized_dataset = raw_dataset.map(
                tokenize_function, 
                batched=True, 
                remove_columns=raw_dataset.column_names
            )
            tokenized_dataset.set_format("torch")

            # 5. Split into Train/Test (90/10)
            split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split['train']
            test_dataset = split['test']
            train_dataset = train_dataset.select(range(1000))
            test_dataset = test_dataset.select(range(100))

            # 6. CRITICAL COMPATIBILITY FIX:
            # Your server uses .targets to split data among clients.
            # We assign random integers [0-9] so the distribution logic works.
            train_dataset.targets = torch.randint(0, 4, (len(train_dataset),))
            test_dataset.targets = torch.randint(0, 4, (len(test_dataset),))

            print(f"[SUCCESS] Loaded Alpaca: {len(train_dataset)} train, {len(test_dataset)} test samples.")
            return train_dataset, test_dataset
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        return train_dataset, test_dataset

    def _split(self, arr, n):
        """
        Helper function: Splits arr into n parts as evenly as possible.
        """
        k, m = divmod(len(arr), n)
        return (arr[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n))
    
    def _flatten_single_gradient(self, grad_dict):
        return torch.cat([v.view(-1) for v in grad_dict.values()])

    def _distribute_dataset(self, train_dataset, num_label, q_factor, batch_size):
        """
        Distributes the dataset among clients in a partially overlapping manner 
        controlled by q_factor. 
        """
        # if self.num_clients < num_label:
        #     raise ValueError("Number of clients must be >= number of classes.")
        num_label = min(num_label, self.num_clients)

        label2idx = {}
        for i in range(num_label):
            label2idx[str(i)] = np.where(train_dataset.targets == i)[0]

        # Each label i is also one group i
        group2client_idx = []
        for i in range(num_label):
            group2client_idx.extend([i] * (self.num_clients // num_label))

        remainder = self.num_clients - len(group2client_idx)
        if remainder > 0:
            # Random assignment of leftover clients
            extra_groups = list(np.random.permutation(num_label)[:remainder])
            group2client_idx.extend(extra_groups)

        random.shuffle(group2client_idx)

        # Build group -> data index
        group2data_idx = [[] for _ in range(num_label)]
        for i in range(num_label):
            perm_idx = np.random.permutation(len(label2idx[str(i)]))
            split_point = int(len(perm_idx) * q_factor)
            # Indices for the group i
            q_group_idx = perm_idx[:split_point]
            q_complement_idx = perm_idx[split_point:]

            group2data_idx[i].extend(label2idx[str(i)][q_group_idx.tolist()])

            # Distribute complement among other groups
            if len(q_complement_idx) > 0:
                splitted_complement = list(self._split(q_complement_idx, num_label - 1))
                c_idx = 0
                for j in range(num_label):
                    if j != i:
                        group2data_idx[j].extend(label2idx[str(i)][splitted_complement[c_idx].tolist()])
                        c_idx += 1

        # Shuffle each group
        for sublist in group2data_idx:
            random.shuffle(sublist)

        # Each group has some subset of data. Now map group -> clients
        client2data_idx = {}
        for i in range(num_label):
            client_indices = np.where(np.array(group2client_idx) == i)[0]
            data_indices = group2data_idx[i]
            splitted_data = list(self._split(data_indices, len(client_indices)))
            for j, c in enumerate(client_indices):
                client2data_idx[c] = splitted_data[j]

        client_loaders = []
        for i in range(self.num_clients):
            subset_indices = client2data_idx[i]
            subset_ds = torch.utils.data.Subset(train_dataset, subset_indices)
            loader = torch.utils.data.DataLoader(subset_ds, batch_size=batch_size, shuffle=True)
            client_loaders.append(loader)

        return client_loaders, group2client_idx

    def _flatten_tensors(self, input_list):
        """
        Flattens each client's list/dict of tensors into a single 1D vector
        and stacks them into a 2D tensor (columns = clients).
        """
        flattened = []
        for tensors in input_list:
            # Each element in `input_list` is typically a list of grads or a dict
            if isinstance(tensors, dict):
                # Sort keys for consistency
                sorted_keys = sorted(tensors.keys())
                cat = torch.cat([tensors[k].view(-1) for k in sorted_keys])
            else:
                cat = torch.cat([t.view(-1) for t in tensors])
            flattened.append(cat)
        return torch.stack(flattened).T

    def _normalize_gradients(self, client_gradients):
        # Compute the norm of each gradient
        grad_norms = [torch.norm(torch.cat([g.view(-1) for g in grad])).item() for grad in client_gradients]
        
        # Compute the median of the norms
        median_norm = np.median(grad_norms)
        
        # Scale each gradient to have the same norm as the median
        for i, grad in enumerate(client_gradients):
            grad_norm = grad_norms[i]
            if grad_norm > 0:
                scale_factor = median_norm / grad_norm
                for j in range(len(grad)):
                    grad[j] = grad[j] * scale_factor

    def _normalize_losses(self, client_losses):
        # Compute the median of the losses
        median_loss = np.median(client_losses)
        
        # Scale each loss to have the same value as the median
        client_losses = [loss * (median_loss / loss) if loss > 0 else loss for loss in client_losses]
        return client_losses

    def _gather_client_updates(
        self,
        global_weights,
        epoch,
        lr,
        return_avg_loss=True,
        compute_gradient=True,
        return_params=False
    ):
        """
        Gathers updates and applies attack + robust defence (mask-based, no client removal).
        """

        client_gradients = []
        client_losses = []

        # ==========================================================
        # ✅ STEP 1: COLLECT ALL CLIENT UPDATES
        # ==========================================================
        for client in self.clients:
            print(f"[Server] Processing Client {client.client_id}", flush=True)

            updates, avg_loss = client.local_update(
                global_weights=global_weights,
                epoch=epoch,
                return_avg_loss=return_avg_loss,
                compute_gradient=compute_gradient,
                return_params=return_params,
                lr=lr,
                server_device=self.device
            )

            client_gradients.append(updates)
            client_losses.append(avg_loss)

        # ==========================================================
        # ✅ STEP 2: REMOVE NaN / INF CLIENTS (HARD FILTER)
        # ==========================================================
        clean_gradients = []
        clean_losses = []
        clean_ids = []

        for i, (g, loss) in enumerate(zip(client_gradients, client_losses)):

            if loss is None or np.isnan(loss) or np.isinf(loss):
                print(f"[Defence] 🚨 Dropping Client {i} (NaN loss)")
                continue

            g_flat = self._flatten_single_gradient(g)

            if torch.isnan(g_flat).any() or torch.isinf(g_flat).any():
                print(f"[Defence] 🚨 Dropping Client {i} (NaN gradient)")
                continue

            clean_gradients.append(g)
            clean_losses.append(loss)
            clean_ids.append(i)

        client_gradients = clean_gradients
        client_losses = clean_losses

        if len(client_gradients) == 0:
            raise RuntimeError("All clients dropped due to NaNs!")

        # ==========================================================
        # ✅ STEP 3: APPLY ATTACK (ONCE)
        # ==========================================================
        if self.attack_func is not None:
            client_gradients = self.attack_func(
                client_gradients,
                self.clients,
                scale=self.attack_args.get("scale", 1.0)
            )
        weights_mask = torch.ones(len(client_gradients))
        return client_gradients, client_losses, weights_mask

    def calculate_accuracy(self, is_fedavg=False):
        # """
        # Evaluates the model on the test dataset.
        # For FedAvg, typically uses a separate global model 
        # than for other strategies.
        # """
        # model = self.global_model
        # model.eval()

        # loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=128, shuffle=False)
        # total_correct = 0
        # total_samples = 0
        # total_loss = 0.0
        # loss_fn = nn.CrossEntropyLoss()

        # with torch.no_grad():
        #     for X, y in loader:
        #         X, y = X.to(self.device), y.to(self.device)
        #         out = model(X)
        #         loss = loss_fn(out, y)

        #         _, preds = torch.max(out, dim=1)
        #         total_correct += (preds == y).sum().item()
        #         total_samples += y.size(0)
        #         total_loss += loss.item()

        # accuracy = 100.0 * total_correct / total_samples
        # avg_loss = total_loss / len(loader)

        # # Check for divergence
        # if np.isnan(avg_loss) or avg_loss > 9:
        #     raise RuntimeError(f"Model diverged at epoch: Loss = {avg_loss}")

        # print(f"Test Accuracy = {accuracy:.2f}%, Test Loss: {avg_loss:.4f}")
        # return accuracy, avg_loss
        loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=8, shuffle=False)

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():

            for batch in loader:

                # Alpaca batches are dicts
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.global_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )

                loss = outputs.loss

                total_loss += loss.item()
                total_tokens += 1

        avg_loss = total_loss / total_tokens

        print(f"Test Loss: {avg_loss:.4f}")

        return 0.0, avg_loss
    
    def _flatten_single_gradient(self, grad_dict):
        return torch.cat([v.view(-1) for v in grad_dict.values()])

