""" Attack Schemes """
import torch
from scipy.stats import norm
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attack:
    def __init__(self, attack_args):

        self.attack_args = attack_args if attack_args is not None else {}
        attack_type      = self.attack_args.get('attack_type', None)
        self.attack_type = attack_type

        self.attack_map = {
            'flip_labels':        self.flip_labels,
            'backdoor':           self.backdoor,
            'random_parameters':  self.random_parameters,
            'boost_gradient':     self.boost_gradient,
            'lie_attack':         self.lie_attack,
            'inverse_gradient':   self.inverse_gradient_attack,
        }

        if attack_type is None or attack_type == "none":
            self.func = self.no_attack
        elif attack_type in self.attack_map:
            self.func = self.attack_map[attack_type]
        else:
            raise Exception(f"Attack type '{attack_type}' is invalid.")

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs, **self.attack_args)

    # ===================================================================
    # DATA ATTACKS (applied during training, not here)
    # ===================================================================

    def flip_labels(*args, **kwargs):
        return kwargs['data'], kwargs['max_label'] - kwargs['target']

    def backdoor(*args, **kwargs):
        backdoor_pattern = kwargs.get('backdoor_pattern', None)
        backdoor_target  = kwargs.get('backdoor_target', None)
        if backdoor_pattern is None:
            raise ValueError("backdoor_pattern is required.")

        i, j, h, w, v = (backdoor_pattern['i'], backdoor_pattern['j'],
                         backdoor_pattern['h'], backdoor_pattern['w'],
                         backdoor_pattern['v'])
        C = kwargs['data'].shape[1]

        if isinstance(v, (int, float)):
            kwargs['data'][:, :, i:i+h, j:j+w] = v
        elif isinstance(v, torch.Tensor) and v.shape == (C,):
            kwargs['data'][:, :, i:i+h, j:j+w] = v.view(C, 1, 1)
        elif isinstance(v, list) and len(v) == C:
            kwargs['data'][:, :, i:i+h, j:j+w] = torch.tensor(v, device=device).view(C, 1, 1)
        else:
            raise ValueError("Invalid value for 'v'.")

        if backdoor_target is not None:
            if backdoor_target == "random":
                kwargs['target'] = torch.randint(
                    kwargs.get('min_label', 0), kwargs['max_label'],
                    size=kwargs['target'].size(), device=device
                )
            else:
                kwargs['target'].fill_(backdoor_target)

        return kwargs['data'], kwargs['target']

    # ===================================================================
    # PARAMETER ATTACK
    # ===================================================================

    def random_parameters(*args, **kwargs):
        mean_factor  = kwargs.get('random_parameters_mean', 0)
        std_factor   = kwargs.get('random_parameters_std', 2)
        global_weights = kwargs['global_weights']

        lora_weights = {k: v for k, v in global_weights.items() if 'lora_' in k}
        target_dict  = lora_weights if len(lora_weights) > 0 else global_weights

        original_shapes     = {name: param.shape for name, param in target_dict.items()}
        concatenated_weights = torch.cat([param.view(-1) for param in target_dict.values()]).to(device)

        std_concat  = torch.std(concatenated_weights)
        mean_concat = torch.mean(concatenated_weights)

        additive_noise = torch.normal(
            mean=-1 * mean_factor * mean_concat,
            std=std_factor * std_concat,
            size=concatenated_weights.shape,
            device=device
        )
        noisy_weights = concatenated_weights + additive_noise

        split_sizes   = [param.numel() for param in target_dict.values()]
        split_tensors = torch.split(noisy_weights, split_sizes)

        attacked_weights = global_weights.copy()
        for i, (name, shape) in enumerate(original_shapes.items()):
            attacked_weights[name] = split_tensors[i].view(shape).to(device)

        return attacked_weights

    # ===================================================================
    # GRADIENT ATTACKS
    # ===================================================================

    def boost_gradient(*args, **kwargs):
        return [kwargs['boost_factor'] * grad for grad in kwargs['grads']]

    def inverse_gradient_attack(self, grads, clients=None, scale=10.0, **kwargs):
        attacked = []
        for g, client in zip(grads, clients):
            if client.malicious:
                new_g = {k: -scale * g[k] + 0.01 * torch.sign(g[k]) for k in g}
                attacked.append(new_g)
            else:
                attacked.append(g)
        return attacked

    def no_attack(self, grads, clients=None, **kwargs):
        return grads

    # ===================================================================
    # LIE ATTACK — fixed for dict-based LoRA gradients
    # ===================================================================
    def lie_attack(self, grads, clients, losses=None, z=None, **kwargs):
        """
        A Little Is Enough (Baruch et al. 2019).

        Crafts malicious gradients that are statistically indistinguishable
        from the benign distribution — staying within z standard deviations
        of the benign mean to evade detection.

        Works with dict-based gradients (LoRA key → tensor).
        """
        num_clients = len(clients)

        # ------------------------------------------------------------------
        # 1. Separate benign gradients and losses
        # ------------------------------------------------------------------
        benign_grads  = [grads[i]  for i, c in enumerate(clients) if not c.malicious]
        benign_losses = [losses[i] for i, c in enumerate(clients) if not c.malicious] \
                        if losses is not None else None

        if len(benign_grads) == 0:
            print("[LIE] No benign clients found — returning original grads")
            return grads

        # ------------------------------------------------------------------
        # 2. Compute z (how many std devs to shift)
        #    Default: derived from the number of malicious clients
        # ------------------------------------------------------------------
        if z is None:
            num_malicious = sum(1 for c in clients if c.malicious)
            s         = num_clients // 2 + 1 + num_malicious
            phi_value = (num_clients - s) / num_clients
            # clamp phi so ppf doesn't blow up
            phi_value = float(np.clip(phi_value, 1e-6, 1 - 1e-6))
            z = norm.ppf(phi_value)
        z = abs(z)

        print(f"[LIE] z = {z:.4f} | benign clients = {len(benign_grads)} | "
              f"malicious clients = {num_clients - len(benign_grads)}")

        # ------------------------------------------------------------------
        # 3. Compute mean + std over benign gradients for each LoRA key
        # ------------------------------------------------------------------
        lora_keys = sorted(benign_grads[0].keys())

        crafted_grad = {}
        for key in lora_keys:
            # Stack benign tensors: shape (num_benign, *param_shape)
            stacked = torch.stack([g[key].float() for g in benign_grads], dim=0)
            mean_t  = stacked.mean(dim=0)
            std_t   = stacked.std(dim=0)

            # LIE: shift mean by z * std toward making aggregation worse
            # sign=+1 pushes in the direction that degrades the model most
            crafted_grad[key] = (mean_t + z * std_t).to(benign_grads[0][key].dtype)

        # ------------------------------------------------------------------
        # 4. Craft malicious loss (shift benign loss mean similarly)
        # ------------------------------------------------------------------
        crafted_loss = None
        if benign_losses is not None:
            mean_loss = float(np.mean(benign_losses))
            std_loss  = float(np.std(benign_losses))
            crafted_loss = mean_loss + z * std_loss

        # ------------------------------------------------------------------
        # 5. Replace malicious client gradients + losses
        # ------------------------------------------------------------------
        attacked_grads  = list(grads)
        attacked_losses = list(losses) if losses is not None else None

        for i, client in enumerate(clients):
            if client.malicious:
                attacked_grads[i] = crafted_grad
                if attacked_losses is not None and crafted_loss is not None:
                    attacked_losses[i] = crafted_loss

        if attacked_losses is not None:
            return attacked_grads, attacked_losses

        return attacked_grads