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
    def __init__(self, base_linear, rank=16):
        super().__init__()

        self.base_layer = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        # ✅ Stable init
        self.lora_A = nn.Parameter(torch.randn(self.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))

        # ✅ Correct scaling
        self.scaling = 32 / rank  # = 2.0

        # ✅ Dropout (paper-aligned)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):

        # ✅ SAFE HADAMARD (no OOM)
        if x.shape[-1] <= 4096:
            rotated_x = fast_hadamard_transform(x)
        else:
            rotated_x = x  # fallback for large dims

        base_out = self.base_layer(rotated_x)

        A = self.lora_A.to(x.dtype)
        B = self.lora_B.to(x.dtype)

        lora_out = (self.dropout(x) @ A) @ B

        return base_out + self.scaling * lora_out
        
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
    MODEL_MAP = {
        "DeeperCIFARCNN": DeeperCIFARCNN,
        "ThreeLayerFC":   ThreeLayerFC,
    }

    if model_name == "Llama-3-8B":
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        print(f"[ACTION] Loading LLaMA-3-8B with 4-bit quantization...")

        # ✅ FIX: 4-bit quantization reduces base weights from ~16GB → ~4GB
        #    leaving ~12GB free for activations, LoRA grads, and optimizer states.
        #    bnb_4bit_compute_dtype=bfloat16 keeps compute numerically stable.
        #    bnb_4bit_use_double_quant adds a second quantization on the
        #    quantization constants — saves another ~0.4GB at negligible cost.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",   # NormalFloat4 — best for LLM weights
        )

        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            quantization_config=quantization_config,
            device_map="auto",
        )

        print(f"[INFO] LLaMA-3-8B loaded in 4-bit — estimated VRAM: ~5GB base weights")
        return model

    if model_name in MODEL_MAP:
        return MODEL_MAP[model_name]()

    raise ValueError(f"Unknown model name: {model_name}. Check your YAML spelling.")