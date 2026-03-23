# # """ Client Class """
# # import torch
# # import torch.nn as nn

# # from attack.attack import Attack


# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None  # Use GradScaler only for CUDA


# # class Client:
# #     ATTACK_ON_DATA = ['flip_labels', 'backdoor']
# #     ATTACK_ON_PARAMETRS = ['random_parameters']
# #     ATTACK_ON_GRADIENT = ['boost_gradient', 'gaussian_attack', 'gaussian_additive_attack']

# #     def __init__(self, client_id, model, data_loader, local_epoch=1, malicious=False, attack_args=None):
# #         self.client_id = client_id
# #         self.model = model
# #         self.data_loader = data_loader
# #         self.malicious = malicious
# #         self.local_epoch = local_epoch

# #         if malicious and attack_args is None:
# #             raise Exception("attack_args is not provided.")

# #         if attack_args is not None:
# #             self.attack_args = attack_args
# #             self.attack_type = attack_args['attack_type']
# #             self.attack_epoch = attack_args['attack_epoch']
# #             self.attack_func = Attack(attack_args)

# #     def local_update(self, global_weights, epoch, return_avg_loss=True, compute_gradient=True, return_params=False, lr=1e-3, server_device=torch.device("cpu")):
# #         local_model = type(self.model)().to(device)
# #         local_model.load_state_dict(global_weights)
# #         local_model.train()

# #         is_under_attack = self.malicious and epoch >= self.attack_epoch
# #         optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

# #         # Attack on Parameters
# #         if is_under_attack and self.attack_type in self.ATTACK_ON_PARAMETRS:
# #             global_weights_random = self.attack_func(global_weights=global_weights, **self.attack_args)
# #             local_model.load_state_dict(global_weights_random)

# #         for local_ep in range(self.local_epoch):
# #             total_loss = 0
# #             num_batches = 0

# #             for data, target in self.data_loader:
# #                 data, target = data.to(device), target.to(device)

# #                 # Attack on Data
# #                 if is_under_attack and self.attack_type in self.ATTACK_ON_DATA:
# #                     # If the client is malicious and the current epoch >= attack_epoch, apply attack on input data
# #                     data, target = self.attack_func(data=data, target=target, **self.attack_args)

# #                 with torch.amp.autocast(device_type=device.type):
# #                     output = local_model(data)
# #                     loss = nn.CrossEntropyLoss()(output, target)

# #                 optimizer.zero_grad()
# #                 if device.type == 'cuda':
# #                     # Use GradScaler for GPU
# #                     scaler.scale(loss).backward()
# #                     scaler.step(optimizer)
# #                     scaler.update()
# #                 else:
# #                     # Standard backward pass for CPU
# #                     loss.backward()
# #                     optimizer.step()

# #                 total_loss += loss.item()
# #                 num_batches += 1

# #                 if not compute_gradient:
# #                     break

# #             if local_ep == self.local_epoch - 1 or not compute_gradient:
# #                 avg_loss = total_loss / num_batches if return_avg_loss else None

# #             if not compute_gradient:
# #                 break

# #         # Move only the state_dict to device
# #         state_dict = {key: value.to(server_device) for key, value in local_model.state_dict().items()}

# #         if return_params:
# #             # Compute parameter updates only for trainable parameters
# #             params = [
# #                 (state_dict[key] - global_weights[key]) for key in global_weights.keys()
# #             ]

# #             # Attack on Gradient
# #             if is_under_attack and self.attack_type in self.ATTACK_ON_GRADIENT:
# #                 params = self.attack_func(grads=params, **self.attack_args)

# #             params = {key: params[i] for i, key in enumerate(global_weights.keys())}
# #         else:
# #             # Get the keys for trainable parameters only
# #             trainable_keys = [name for name, _ in local_model.named_parameters()]

# #             # Compute parameter updates only for trainable parameters
# #             params = [
# #                 -1 * (state_dict[key] - global_weights[key]) / lr
# #                 for key in trainable_keys
# #             ]

# #             # Attack on Gradient
# #             if is_under_attack and self.attack_type in self.ATTACK_ON_GRADIENT:
# #                 params = self.attack_func(grads=params, **self.attack_args)

# #         del local_model, optimizer

# #         return params, avg_loss
    
# #     def local_train(model, train_loader, round_num):
# #     # RoLoRA Strategy: Alternate training A and B by round
# #     # This prevents the 'saturation' seen in FFA-LoRA
# #         for name, param in model.named_parameters():
# #             if 'lora_A' in name:
# #                 param.requires_grad = (round_num % 2 != 0) # Update A on odd rounds
# #             if 'lora_B' in name:
# #                 param.requires_grad = (round_num % 2 == 0) # Update B on even rounds

# #     # Standard training loop follows...

# import torch
# import torch.nn as nn
# from attack.attack import Attack
# import copy

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None

# class Client:
#     ATTACK_ON_DATA = ['flip_labels', 'backdoor']
#     ATTACK_ON_PARAMETRS = ['random_parameters']
#     ATTACK_ON_GRADIENT = ['boost_gradient', 'gaussian_attack', 'gaussian_additive_attack']

#     # def __init__(self, client_id, model, data_loader, local_epoch=1, malicious=False, attack_args=None):
#     #     self.client_id = client_id
#     #     self.model = model
#     #     self.data_loader = data_loader
#     #     self.malicious = malicious
#     #     self.local_epoch = local_epoch

#     #     # --- FIX: Default initialization to prevent attribute errors ---
#     #     self.attack_args = attack_args
#     #     self.attack_type = "none"
#     #     self.attack_epoch = float('inf')
#     #     self.attack_func = None

#     #     if malicious and attack_args is None:
#     #         raise Exception("attack_args is not provided.")

#     #     if attack_args is not None:
#     #         self.attack_type = attack_args.get('attack_type', 'none')
#     #         self.attack_epoch = attack_args.get('attack_epoch', 0)
#     #         self.attack_func = Attack(attack_args)
#     def __init__(self, client_id, model, data_loader, local_epoch=1, malicious=False, attack_args=None):
#             self.client_id = client_id
#             self.model = model
#             self.data_loader = data_loader
#             self.malicious = malicious
#             self.local_epoch = local_epoch

#             # Set safe defaults
#             self.attack_args = attack_args if attack_args is not None else {}
#             self.attack_type = "none"
#             self.attack_epoch = float('inf')
#             self.attack_func = None

#             # Self-healing logic: 
#             # If malicious but no args, downgrade to benign instead of crashing.
#             if self.malicious and attack_args is None:
#                 print(f"⚠️ [WARNING] Client {self.client_id} marked as malicious but attack_args is missing. Reverting to benign.")
#                 self.malicious = False

#             # If we have args, populate the attack settings
#             if attack_args is not None:
#                 from attack.attack import Attack # Local import to avoid circular dependencies
#                 self.attack_type = attack_args.get('attack_type', 'none')
#                 # Use 0 as default if not specified so attack starts immediately
#                 self.attack_epoch = attack_args.get('attack_epoch', 0)
#                 self.attack_func = Attack(attack_args)
# def local_update(
#     self,
#     global_weights,
#     epoch,
#     return_avg_loss=True,
#     compute_gradient=True,
#     return_params=False,
#     lr=1e-3,
#     server_device=torch.device("cpu")
# ):

#     import copy
#     torch.cuda.empty_cache()

#     # ---------------------------------------------------------
#     # MODEL INITIALIZATION (LLM safe)
#     # ---------------------------------------------------------
#     is_llm = hasattr(self.model, "config") and "llama" in str(self.model.config.model_type).lower()

#     if is_llm:
#         # DO NOT duplicate huge LLM model
#         local_model = self.model
#     else:
#         # CNN / small models still safe to copy
#         local_model = copy.deepcopy(self.model)

#     local_model = local_model.to(device)
#     local_model.load_state_dict(global_weights)
#     local_model.train()

#     # ---------------------------------------------------------
#     # RoLoRA DETECTION
#     # ---------------------------------------------------------
#     all_param_names = [name for name, _ in local_model.named_parameters()]
#     has_lora = any("lora_" in name for name in all_param_names)

#     if has_lora:
#         for name, param in local_model.named_parameters():

#             if "lora_A" in name:
#                 param.requires_grad = (epoch % 2 != 0)

#             elif "lora_B" in name:
#                 param.requires_grad = (epoch % 2 == 0)

#             else:
#                 param.requires_grad = False

#     # ---------------------------------------------------------
#     # ATTACK SETUP
#     # ---------------------------------------------------------
#     is_under_attack = self.malicious and epoch >= self.attack_epoch

#     trainable_params = [p for p in local_model.parameters() if p.requires_grad]

#     optimizer = torch.optim.SGD(trainable_params, lr=lr)

#     # Attack on parameters
#     if is_under_attack and self.attack_type in self.ATTACK_ON_PARAMETRS:
#         global_weights_random = self.attack_func(global_weights=global_weights, **self.attack_args)
#         local_model.load_state_dict(global_weights_random)

#     # ---------------------------------------------------------
#     # TRAINING LOOP
#     # ---------------------------------------------------------
#     total_loss = 0
#     num_batches = 0

#     for local_ep in range(self.local_epoch):

#         for batch in self.data_loader:

#             # ---- Dataset compatibility (CNN vs LLM)
#             if isinstance(batch, dict):

#                 data = batch["input_ids"].to(device)
#                 target = batch.get("labels", data).to(device)

#             else:

#                 data = batch[0].to(device)
#                 target = batch[1].to(device)

#             # ---- Attack on data
#             if is_under_attack and self.attack_type in self.ATTACK_ON_DATA:
#                 data, target = self.attack_func(data=data, target=target, **self.attack_args)

#             # ---- Forward pass
#             with torch.amp.autocast(device_type=device.type):

#                 if isinstance(batch, dict):
#                     output = local_model(data, labels=target)
#                     loss = output.loss
#                 else:
#                     output = local_model(data)
#                     loss = nn.CrossEntropyLoss()(output, target)

#             optimizer.zero_grad()

#             if device.type == "cuda" and scaler is not None:

#                 scaler.scale(loss).backward()
#                 scaler.step(optimizer)
#                 scaler.update()

#             else:

#                 loss.backward()
#                 optimizer.step()

#             total_loss += loss.item()
#             num_batches += 1

#             if not compute_gradient:
#                 break

#         if not compute_gradient:
#             break

#     avg_loss = total_loss / max(num_batches, 1)

#     # ---------------------------------------------------------
#     # PARAMETER COLLECTION (SparseFL compatible)
#     # ---------------------------------------------------------
#     state_dict = {k: v.detach().to(server_device) for k, v in local_model.state_dict().items()}

#     trainable_keys = [name for name, p in local_model.named_parameters() if p.requires_grad]

#     target_keys = all_param_names

#     params = []

#     for key in target_keys:

#         if key in trainable_keys:

#             delta = -1 * (state_dict[key] - global_weights[key].to(server_device)) / lr
#             params.append(delta)

#         else:

#             params.append(torch.zeros_like(global_weights[key], device=server_device))

#     # ---------------------------------------------------------
#     # ATTACK ON GRADIENT
#     # ---------------------------------------------------------
#     if is_under_attack and self.attack_type in self.ATTACK_ON_GRADIENT:
#         params = self.attack_func(grads=params, **self.attack_args)

#     del optimizer
#     torch.cuda.empty_cache()

#     return params, avg_loss

#     def local_train(self, model, train_loader, round_num):
#         # Kept for backward compatibility with your existing calls
#         for name, param in model.named_parameters():
#             if 'lora_A' in name:
#                 param.requires_grad = (round_num % 2 != 0)
#             if 'lora_B' in name:
#                 param.requires_grad = (round_num % 2 == 0)
import torch
import torch.nn as nn
import copy
from attack.attack import Attack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None


class Client:

    ATTACK_ON_DATA = ['flip_labels', 'backdoor']
    ATTACK_ON_PARAMETRS = ['random_parameters']
    ATTACK_ON_GRADIENT = ['boost_gradient', 'gaussian_attack', 'gaussian_additive_attack']

    def __init__(self, client_id, model, data_loader, local_epoch=1, malicious=False, attack_args=None):

        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.local_epoch = local_epoch
        self.malicious = malicious

        self.attack_args = attack_args if attack_args is not None else {}
        self.attack_type = "none"
        self.attack_epoch = float("inf")
        self.attack_func = None

        if self.malicious and attack_args is None:
            print(f"⚠️ Client {client_id} malicious but attack_args missing. Switching to benign.", flush=True)
            self.malicious = False

        if attack_args is not None:
            self.attack_type = attack_args.get("attack_type", "none")
            self.attack_epoch = attack_args.get("attack_epoch", 0)
            self.attack_func = Attack(attack_args)

    def local_update(
        self,
        global_weights,
        epoch,
        return_avg_loss=True,
        compute_gradient=True,
        return_params=False,
        lr=1e-3,
        server_device=torch.device("cpu")
    ):

        print(f"\n[Client {self.client_id}] Starting local_update (round {epoch})", flush=True)

        if torch.cuda.is_available():
            print(f"[Client {self.client_id}] GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

        torch.cuda.empty_cache()

        # ==========================================================
        # ORIGINAL MODEL HANDLING
        # ==========================================================
        is_llm = hasattr(self.model, "config") and "llama" in str(self.model.config.model_type).lower()

        if is_llm:
            local_model = self.model
            print(f"[Client {self.client_id}] Using shared LLM model", flush=True)
        else:
            local_model = copy.deepcopy(self.model)
            print(f"[Client {self.client_id}] Using copied CNN model", flush=True)

        local_model = local_model.to(device)
        local_model.load_state_dict(global_weights)
        local_model.train()

        print(f"[Client {self.client_id}] Model loaded to {device}", flush=True)

        # ==========================================================
        # LoRA detection
        # ==========================================================
        all_param_names = [name for name, _ in local_model.named_parameters()]
        has_lora = any("lora_" in name for name in all_param_names)

        print(f"[Client {self.client_id}] RoLoRA detected: {has_lora}", flush=True)

        # ==========================================================
        # ORIGINAL LoRA alternating training
        # ==========================================================
        if has_lora:
            for name, param in local_model.named_parameters():

                if "lora_A" in name:
                    param.requires_grad = (epoch % 2 != 0)

                elif "lora_B" in name:
                    param.requires_grad = (epoch % 2 == 0)

                else:
                    param.requires_grad = False

        is_under_attack = self.malicious and epoch >= self.attack_epoch

        trainable_params = [p for p in local_model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, lr=lr)

        # ==========================================================
        # Parameter attack
        # ==========================================================
        if is_under_attack and self.attack_type in self.ATTACK_ON_PARAMETRS:
            print(f"[Client {self.client_id}] Parameter attack active", flush=True)
            random_weights = self.attack_func(global_weights=global_weights, **self.attack_args)
            local_model.load_state_dict(random_weights)

        total_loss = 0
        num_batches = 0

        print(f"[Client {self.client_id}] Starting local training loop", flush=True)

        # ==========================================================
        # TRAINING LOOP
        # ==========================================================
        for local_ep in range(self.local_epoch):

            print(f"[Client {self.client_id}] Local epoch {local_ep+1}/{self.local_epoch}", flush=True)

            for batch in self.data_loader:

                if not isinstance(batch, dict):
                    data, target = batch[0].to(device), batch[1].to(device)
                else:
                    data = batch["input_ids"].to(device)
                    target = batch.get("labels", data).to(device)

                if num_batches == 0:
                    print(f"[Client {self.client_id}] First forward pass", flush=True)

                # Data attack
                if is_under_attack and self.attack_type in self.ATTACK_ON_DATA:
                    data, target = self.attack_func(data=data, target=target, **self.attack_args)

                with torch.amp.autocast(device_type=device.type):

                    if isinstance(batch, dict):
                        output = local_model(data, labels=target)
                        loss = output.loss
                    else:
                        output = local_model(data)
                        loss = nn.CrossEntropyLoss()(output, target)

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[Client {self.client_id}] 🚨 NaN loss detected — stopping early")
                        break

                optimizer.zero_grad()

                if device.type == "cuda" and scaler is not None:

                    scaler.scale(loss).backward()

                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)

                    scaler.step(optimizer)
                    scaler.update()

                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if not compute_gradient:
                    break

            if not compute_gradient:
                break

        avg_loss = total_loss / max(num_batches, 1)

        print(f"[Client {self.client_id}] Training complete | avg_loss={avg_loss:.4f}", flush=True)

        print(f"[Client {self.client_id}] Collecting parameter updates", flush=True)

        # ==========================================================
        # ORIGINAL PARAM UPDATE (WORKING VERSION)
        # ==========================================================
        state_dict = {k: v.detach().cpu() for k, v in local_model.state_dict().items()}
        trainable_keys = [name for name, p in local_model.named_parameters() if p.requires_grad]

        if has_lora:
            target_keys = [k for k in state_dict.keys() if "lora_" in k]
        else:
            target_keys = list(state_dict.keys())

        params = {}

        for key in target_keys:

            if key in trainable_keys:
                delta = -1 * (state_dict[key] - global_weights[key].cpu()) / lr
                params[key] = delta
            else:
                params[key] = torch.zeros_like(global_weights[key]).cpu()

        # ==========================================================
        # Gradient attack
        # ==========================================================
        if is_under_attack and self.attack_type in self.ATTACK_ON_GRADIENT:
            print(f"[Client {self.client_id}] Gradient attack applied", flush=True)
            params = self.attack_func(grads=params, **self.attack_args)

        del optimizer
        torch.cuda.empty_cache()

        print(f"[Client {self.client_id}] Finished local_update", flush=True)

        return params, avg_loss