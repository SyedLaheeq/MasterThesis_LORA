import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Client:
    def __init__(
        self,
        client_id,
        model,
        train_dataset,
        device="cuda",
        batch_size=4,
        local_epochs=1,
        malicious=False,
        attack_args=None,
        learning_rate=1e-4,
    ):
        self.client_id     = client_id
        self.model         = model
        self.train_dataset = train_dataset
        self.device        = device
        self.batch_size    = batch_size
        self.local_epochs  = local_epochs
        self.malicious     = malicious
        self.attack_args   = attack_args
        self.learning_rate = learning_rate

    def local_update(
        self,
        global_weights,
        epoch,
        lr=None,
        return_avg_loss=True,
        compute_gradient=True,
        return_params=False,
        server_device="cuda",
    ):
        print(f"\n========== CLIENT {self.client_id} ROUND {epoch} ==========")

        effective_lr = float(lr) if lr is not None else float(self.learning_rate)

        self._set_model_weights(global_weights)

        # Do NOT call model.to(device) — 4-bit quant + device_map="auto"
        # manages placement. Calling .to() breaks it.
        self.model.train()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            print(f"[Client {self.client_id}] WARNING: No trainable parameters found!")
            return None, None

        print(f"[Client {self.client_id}] Trainable params : {len(trainable_params)}")
        print(f"[Client {self.client_id}] LR               : {effective_lr}")

        # 8-bit AdamW: optimizer states stored in int8 instead of float32
        # saves ~75% of optimizer state memory (~300MB for 419 LoRA tensors)
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=effective_lr,
                weight_decay=0.01,
            )
            print(f"[Client {self.client_id}] Optimizer: 8-bit AdamW")
        except ImportError:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=effective_lr,
                weight_decay=0.01,
            )
            print(f"[Client {self.client_id}] Optimizer: standard AdamW (install bitsandbytes for 8-bit)")

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        total_loss  = 0.0
        num_batches = 0

        for _ in range(self.local_epochs):
            for batch in train_loader:

                input_ids = batch["input_ids"]
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids)

                attention_mask = batch["attention_mask"]
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(attention_mask)

                target_device  = next(self.model.parameters()).device
                input_ids      = input_ids.to(target_device)
                attention_mask = attention_mask.to(target_device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )
                    loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[Client {self.client_id}] NaN/Inf loss — skipping batch")
                    torch.cuda.empty_cache()
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                total_loss  += loss.item()
                num_batches += 1

                if num_batches >= 50:
                    break

            if num_batches >= 50:
                break

        # Free optimizer states immediately after each client finishes
        del optimizer
        torch.cuda.empty_cache()

        if num_batches == 0:
            print(f"[Client {self.client_id}] No valid batches — returning None")
            return None, None

        avg_loss = total_loss / num_batches

        if compute_gradient:
            updates = self._get_lora_gradients(global_weights)
        else:
            updates = self._get_lora_params()

        if updates is None or len(updates) == 0:
            print(f"[Client {self.client_id}] WARNING: Empty updates!")
            return None, None

        norm = self._compute_update_norm(updates)
        print(f"[Client {self.client_id}] Update norm : {norm:.6f}")
        print(f"[Client {self.client_id}] Avg loss    : {avg_loss:.4f}")
        print(f"[Client {self.client_id}] Malicious   : {self.malicious}")
        print("========== CLIENT DONE ==========")

        return updates, avg_loss

    def _set_model_weights(self, global_weights):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_weights:
                    param.data.copy_(global_weights[name].to(param.device))

    def _get_lora_gradients(self, global_weights):
        updates = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "lora_" in name and name in global_weights:
                    delta = param.data - global_weights[name].to(param.device)
                    updates[name] = delta.detach().cpu().clone()
        if len(updates) == 0:
            print(f"[Client {self.client_id}] WARNING: No LoRA params found!")
        return updates

    def _get_lora_params(self):
        params = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    params[name] = param.data.detach().cpu().clone()
        return params

    def _compute_update_norm(self, updates):
        total_norm = 0.0
        for v in updates.values():
            total_norm += torch.norm(v.float()).item() ** 2
        return total_norm ** 0.5