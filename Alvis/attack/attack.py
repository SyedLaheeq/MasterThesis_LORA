""" Attack Schemes """
import torch
from scipy.stats import norm
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attack:
    def __init__(self, attack_args):

        self.attack_args = attack_args if attack_args is not None else {}

        attack_type = self.attack_args.get('attack_type', None)
        self.attack_type = attack_type

        # -------------------------------
        # Attack mapping
        # -------------------------------
        self.attack_map = {

            # Data attacks
            'flip_labels': self.flip_labels,
            'backdoor': self.backdoor,

            # Parameter attacks
            'random_parameters': self.random_parameters,

            # Gradient attacks
            'boost_gradient': self.boost_gradient,
            'lie_attack': self.lie_attack,

            # 🔥 NEW
            'inverse_gradient': self.inverse_gradient_attack,
        }

        if attack_type is None or attack_type == "none":
            self.func = self.no_attack

        elif attack_type in self.attack_map:
            self.func = self.attack_map[attack_type]

        else:
            raise Exception(f"Attack type '{attack_type}' is invalid.")

    # -------------------------------
    # CALL
    # -------------------------------
    def __call__(self, **kwargs):
        return self.func(**kwargs)

    # -------------------------------
    # DEFAULT (no attack)
    # -------------------------------
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs, **self.attack_args)
    
    def flip_labels(*args, **kwargs):
        return kwargs['data'], kwargs['max_label'] - kwargs['target']

    def backdoor(*args, **kwargs):
        backdoor_pattern = kwargs.get('backdoor_pattern', None)
        backdoor_target = kwargs.get('backdoor_target', None)
        if backdoor_pattern is None:
            raise ValueError("backdoor_pattern is required.")
    
        i, j, h, w, v = backdoor_pattern['i'], backdoor_pattern['j'], backdoor_pattern['h'], backdoor_pattern['w'], backdoor_pattern['v']
        C = kwargs['data'] .shape[1]
        
        # Apply backdoor to the specified region
        if isinstance(v, (int, float)):
            # Single value for all channels
            kwargs['data'] [:, :, i:i+h, j:j+w] = v
        elif isinstance(v, torch.Tensor) and v.shape == (C,):
            # Channel-specific values
            kwargs['data'] [:, :, i:i+h, j:j+w] = v.view(C, 1, 1)
        elif isinstance(v, list) and len(v) == C:
            # Channel-specific values (list)
            kwargs['data'] [:, :, i:i+h, j:j+w] = torch.tensor(v, device=device).view(C, 1, 1)
        else:
            raise ValueError("Invalid value for 'v'. Must be an int, float, or a tensor of shape [C].")
        
        if backdoor_target is not None:
            if backdoor_target == "random":
                kwargs['target'] = torch.randint(kwargs.get('min_label', 0), kwargs['max_label'], size=kwargs['target'].size(), device=device)
            else:
                kwargs['target'].fill_(backdoor_target)

        return kwargs['data'], kwargs['target'] # Number of channels (1 for MNIST, 3 for CIFAR)

    # Attack on Parameters
    def random_parameters(*args, **kwargs):
        mean_factor = kwargs.get('random_parameters_mean', 0)
        std_factor = kwargs.get('random_parameters_std', 2)
        global_weights = kwargs['global_weights']
        
        # Filter: Only attack LoRA weights to save memory and be precise
        lora_weights = {k: v for k, v in global_weights.items() if 'lora_' in k}
        target_dict = lora_weights if len(lora_weights) > 0 else global_weights

        original_shapes = {name: param.shape for name, param in target_dict.items()}
        concatenated_weights = torch.cat([param.view(-1) for param in target_dict.values()]).to(device)

        std_concat = torch.std(concatenated_weights)
        mean_concat = torch.mean(concatenated_weights)

        additive_noise = torch.normal(
            mean=-1 * mean_factor * mean_concat,
            std=std_factor * std_concat,
            size=concatenated_weights.shape,
            device=device
        )
        noisy_weights = concatenated_weights + additive_noise

        split_sizes = [param.numel() for param in target_dict.values()]
        split_tensors = torch.split(noisy_weights, split_sizes)

        # Reconstruct the dict
        attacked_weights = global_weights.copy()
        for i, (name, shape) in enumerate(original_shapes.items()):
            attacked_weights[name] = split_tensors[i].view(shape).to(device)

        return attacked_weights

    # Attack on Gradient
    def boost_gradient(*args, **kwargs):
        return [kwargs['boost_factor'] * grad for grad in kwargs['grads']]


    def inverse_gradient_attack(self, grads, clients=None, scale=1.0, **kwargs):

        attacked = []

        for g in grads:
            new_g = {}

            for k in g:
                new_g[k] = -scale * g[k]

                # amplify small gradients
                if torch.norm(new_g[k]) < 1e-3:
                    new_g[k] = new_g[k] * 100

            attacked.append(new_g)

        return attacked
    
    def no_attack(grads, clients, **kwargs):
        return grads


    def get_attack_function(attack_type):
        if attack_type == "inverse_gradient":
            return inverse_gradient_attack
        else:
            return no_attack
    def lie_attack(*args, **kwargs):
        clients = kwargs['clients']
        clients_grads = kwargs['grads']
        clients_losses = kwargs.get('losses', None)
        num_clients = len(clients)

        clients_params = [clients_grads, clients_losses]

        malicious_gradients = [
            clients_grads[i]
            for i, client in enumerate(clients) if not client.malicious
        ]

        malicious_losses = None
        if clients_losses is not None:
            malicious_losses = [
                clients_losses[i]
                for i, client in enumerate(clients) if not client.malicious
            ]

        malicious_params = [malicious_gradients, malicious_losses]

        s = num_clients // 2 + 1 + num_clients - len(malicious_gradients)
        phi_value = (num_clients - s) / num_clients
        z_default = norm.ppf(phi_value)

        z = kwargs.get("z", z_default)

        # Initialize dictionary to store crafted malicious gradient
        if isinstance(clients_grads[0], dict):
            for_list = clients_grads[0].keys()
            attacked_grad = {}
            attacked_losses = [[]] * len(clients_grads[0])
            attacked_params = [attacked_grad, attacked_losses]
            sign_z = 1
        else:
            for_list = range(len(clients_grads[0]))
            attacked_grad = [[]] * len(clients_grads[0])
            attacked_losses = [[]] * len(clients_grads[0])
            attacked_params = [attacked_grad, attacked_losses]
            sign_z = 1

        # Stack tensors for each key in the gradient dictionaries
        for k, malicious in enumerate(malicious_params):
            if malicious is not None:
                if isinstance(malicious[0], float):
                    mean_list = np.mean(malicious, axis=0)
                    std_list = np.std(malicious, axis=0)
                    attacked_params[k] = float(mean_list + sign_z * z * std_list)
                else:
                    for key in for_list:
                        # Extract tensors for the current key from all malicious gradients
                        stacked_tensors = torch.stack([grad[key].float() for grad in malicious])
                        mean_tensor = torch.mean(stacked_tensors, dim=0)
                        std_tensor = torch.std(stacked_tensors, dim=0)

                        # Craft the malicious gradient for the current key
                        attacked_params[k][key] = mean_tensor + sign_z * z * std_tensor

        # Replace gradients for malicious clients
        for i, client in enumerate(clients):
            if client.malicious:
                for j, param in enumerate(clients_params):
                    if param is not None:
                        param[i] = attacked_params[j]

        return clients_params

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
