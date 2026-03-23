""" Define Models """
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from utils.math_utils import fast_hadamard_transform

# --- Iteration 1 & 2 Models (Baseline & FedLAW) ---

class ThreeLayerFC(nn.Module):
    def __init__(self):
        super(ThreeLayerFC, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DeeperCIFARCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeeperCIFARCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.GroupNorm(num_groups=4, num_channels=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.GroupNorm(num_groups=16, num_channels=128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.GroupNorm(num_groups=32, num_channels=256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool2(torch.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Iteration 4: Rotated RoLoRA for LLMs ---

class RotatedLoRALinear(nn.Module):
    def __init__(self, base_linear, rank=8, scaling=0.01):
        super().__init__()
        self.base_layer = base_linear  # Pre-rotated frozen weights
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        
        # RoLoRA Adapters: Initialized as per paper to maintain stability
        self.lora_A = nn.Parameter(torch.randn(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        self.scaling = scaling

    def forward(self, x):
        # 1. Rotate input to eliminate outliers (Paper Technical Aspect)
        # Uses Walsh-Hadamard Transform to smear 'heavy-hitter' activations
        rotated_x = fast_hadamard_transform(x)
        
        # 2. Base path: uses weights that were rotated during injection
        base_out = self.base_layer(rotated_x)
        
        # 3. LoRA Path: Original input space for gradient stability
        lora_out = (x @ self.lora_A.to(x.dtype)) @ self.lora_B.to(x.dtype)
        
        return base_out + (lora_out * self.scaling)
    
def inject_rolora_to_llama(model, rank=8):
    """
    Transform a LLaMA model into a RoLoRA model safely for 2D or 3D weights.
    1. Permanently rotates frozen weights along the last dimension using Hadamard Transform.
    2. Replaces nn.Linear layers with RotatedLoRALinear.
    """

    target_layers = ['q_proj', 'v_proj', 'k_proj', 'down_proj']

    print(f"[ACTION] Applying RoLoRA: Performing Hadamard Transform and Injecting Adapters (rank={rank})...")

    for name, module in model.named_modules():

        if any(t in name for t in target_layers) and isinstance(module, nn.Linear):

            # --- Step 1: Rotate frozen weights ---
            with torch.no_grad():

                orig_shape = module.weight.shape
                dim = orig_shape[-1]

                w_flat = module.weight.data.reshape(-1, dim)

                rotated_w_flat = fast_hadamard_transform(w_flat)

                new_w = rotated_w_flat.view(orig_shape)

                module.weight.copy_(new_w)

            # 🔹 Freeze base weights (IMPORTANT)
            for p in module.parameters():
                p.requires_grad = False

            # --- Step 2: Replace Linear with RotatedLoRALinear ---
            parent_path = name.rsplit('.', 1)

            if len(parent_path) > 1:

                parent = model.get_submodule(parent_path[0])
                child_name = parent_path[1]

                new_layer = RotatedLoRALinear(module, rank=rank)

                setattr(parent, child_name, new_layer)

            else:

                new_layer = RotatedLoRALinear(module, rank=rank)

                model.add_module(name, new_layer)

    return model

def get_model(model_name):
    """Retrieve model instance by name. Supports CNNs and LLaMA."""
    # Mapping of local model names to classes defined above in this file
    MODEL_MAP = {
        "DeeperCIFARCNN": DeeperCIFARCNN,
        "ThreeLayerFC": ThreeLayerFC,
    }

    # LLaMA-3 Support for Iteration 4 (RoLoRA)
    if model_name == "Llama-3-8B":
        from transformers import AutoModelForCausalLM
        import torch
        print(f"Loading Base LLaMA-3-8B for Outlier-Free Rotation...")
        # Load in half-precision to save VRAM
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            torch_dtype=torch.float16,
            device_map="auto" 
        )
        return model

    if model_name in MODEL_MAP:
        return MODEL_MAP[model_name]()
    else:
        raise ValueError(f"Unknown model name: {model_name}. Check your YAML spelling.")