"""Class for defence"""
import torch
import random
import time
import math


class Defence:
    def __init__(self, defence_args=None):
        defence_type = defence_args.get('defence_type', 'no_defense')

        if defence_type == "no_defence":
            self.func = self.no_defense
        elif defence_type == "krum":
            self.func = self.krum
        elif defence_type == "trimmed_mean":
            self.func = self.trimmed_mean
        elif defence_type == "bulyan":
            self.func = self.bulyan
        elif defence_type == "bulyan_bucketing":
            self.func = self.bulyan_bucketing
        elif defence_type == "cclip":
            self.momentum = None
            self.func = self.cclip
        elif defence_type == "cclip_bucketing":
            self.momentum = None
            self.func = self.cclip_bucketing
        elif defence_type == "rfa":
            self.func = self.rfa
        elif defence_type == "rfa_bucketing":
            self.func = self.rfa_bucketing
        elif defence_type == "huber":
            self.func = self.huber
        elif defence_type == "coord_median":
            self.func = self.coord_median
        else:
            raise ValueError("defence_type is not valid.")

    @staticmethod
    def _flatten_update(update_dict, keys):
        """Concatenate all tensors specified by `keys` into one 1‑D vector."""
        return torch.cat([update_dict[k].flatten() for k in keys], dim=0)

    @staticmethod
    def _unflatten_vector(vector, template_update, keys):
        """Split `vector` back into the original dict‐of‐tensors structure."""
        out, ofs = {}, 0
        for k in keys:
            numel = template_update[k].numel()
            out[k] = vector[ofs : ofs + numel].view_as(template_update[k])
            ofs += numel
        return out

    @staticmethod
    def _clip(v, tau):
        """ℓ₂ centred clipping:  clipτ(u) = u · min(1, τ / ‖u‖₂)."""
        norm = torch.norm(v)
        scale = min(1.0, tau / (norm + 1e-12))
        return v * scale

    def no_defense(self, *args, **kwargs):
        delta_local_updates = kwargs['delta_local_updates']
        keys = delta_local_updates[0].keys()
        num_clients = len(delta_local_updates)

        # Incremental sum approach:
        aggregated = {
            k: torch.zeros_like(delta_local_updates[0][k]) for k in keys
        }

        for update in delta_local_updates:
            for k in keys:
                aggregated[k] += update[k]

        for k in keys:
            aggregated[k] /= num_clients

        return aggregated

    def krum(self, *args, **kwargs):
        delta_local_updates = kwargs['delta_local_updates']
        non_malicious_count = kwargs.get('krum_factor', len(delta_local_updates) - 2)
        return_index = kwargs.get("return_index", False)

        # Infer the device from the first tensor in delta_local_updates
        device = next(iter(delta_local_updates[0].values())).device

        num_updates = len(delta_local_updates)
        if num_updates < 2:
            # Edge case: If there's only one or zero updates, just return it.
            return delta_local_updates[0], 0 if return_index else delta_local_updates[0]


        # 1. Flatten each update into one large vector
        keys = list(delta_local_updates[0].keys())
        dim_per_update = sum(delta_local_updates[0][k].numel() for k in keys)
        
        with torch.no_grad():
            updates_flat = torch.empty((num_updates, dim_per_update), device=device)
            for i, update_dict in enumerate(delta_local_updates):
                # Flatten each tensor & concatenate
                flat_list = [update_dict[k].flatten() for k in keys]
                updates_flat[i] = torch.cat(flat_list, dim=0).to(device)

            # 2. Compute pairwise squared distances via:
            #    dist^2(Xi, Xj) = ||Xi||^2 + ||Xj||^2 - 2 Xi·Xj
            #    This yields the full NxN distance matrix in one shot.
            # -----------------------------------------------------
            # Precompute L2 norms (row-wise)
            norms = (updates_flat ** 2).sum(dim=1, keepdim=True)
            # Expand and compute all pairwise squared distances
            distances = norms + norms.T - 2.0 * (updates_flat @ updates_flat.T)
            # Numerical issues may cause tiny negative values; clamp them to 0
            distances.clamp_(min=0)

            # 3. For each row i, sort distances to find the sum of the closest
            #    `non_malicious_count` neighbors (excluding self-distance)
            sorted_distances, _ = distances.sort(dim=1)
            # Exclude self-distance, which should be the 0 at sorted_distances[:, 0]
            # Then sum up the next `non_malicious_count` distances
            scores = sorted_distances[:, 1 : 1 + non_malicious_count].sum(dim=1)

            # 4. Pick the update with the minimum Krum score
            krum_index = scores.argmin().item()


        if return_index:
            return delta_local_updates[krum_index], krum_index
        else:
            return delta_local_updates[krum_index]

    def trimmed_mean(self, *args, **kwargs):
        delta_local_updates = kwargs['delta_local_updates']
        beta = kwargs.get('trimmed_factor', 0.1)

        num_clients = len(delta_local_updates)
        trimmed_weights = {key: [] for key in delta_local_updates[0].keys()}

        for key in trimmed_weights.keys():
            for client_update in delta_local_updates:
                trimmed_weights[key].append(client_update[key])

            trimmed_weights[key] = torch.stack(trimmed_weights[key])
            sorted_weights, _ = torch.sort(trimmed_weights[key], dim=0)
            lower_bound = int(beta * num_clients)
            upper_bound = num_clients - lower_bound
            trimmed_weights[key] = sorted_weights[lower_bound:upper_bound].mean(dim=0)

        return trimmed_weights

    def bulyan(self, *args, **kwargs):
        delta_local_updates = kwargs['delta_local_updates']
        m = kwargs.get('bulyan_factor', len(delta_local_updates) // 4)
        n = len(delta_local_updates)

        device = next(iter(delta_local_updates[0].values())).device

        import time
        t = time.time()
        if device == torch.device("cpu"):
            # 1) Move everything to GPU (for speed) - if it fits
            device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for update in delta_local_updates:
                for k, v in update.items():
                    update[k] = v.to(device_gpu)

        # Step 1: Perform iterative Krum selection to create a candidate set
        candidate_set = []
        for _ in range(n - 2 * m):
            kwargs['delta_local_updates'] = delta_local_updates
            candidate, krum_index = self.krum(return_index=True, **kwargs)
            candidate = {k: v.cpu() for k, v in candidate.items()}
            candidate_set.append(candidate)
            delta_local_updates.pop(krum_index)

        with torch.no_grad():
            # Step 2: Perform trimmed mean aggregation over the candidate set
            aggregated_weights = {key: [] for key in candidate_set[0].keys()}

            for key in aggregated_weights.keys():
                for candidate in candidate_set:
                    aggregated_weights[key].append(candidate[key])

                aggregated_weights[key] = torch.stack(aggregated_weights[key])
                sorted_weights, _ = torch.sort(aggregated_weights[key], dim=0)
                lower_bound = max(0, min(m, len(candidate_set) // 2))  # Ensure valid lower bound
                upper_bound = max(lower_bound + 1, len(candidate_set) - m)  # Ensure valid range

                aggregated_weights[key] = sorted_weights[lower_bound:upper_bound].float().mean(dim=0)

        print(time.time() - t)
        return aggregated_weights

    def cclip(self, *args, **kwargs):
        """
        Aggregation rule
            m ← m + 1/n Σ_i clipτ( g_i − m ),   repeated `n_iter` times
        where `m` is the running centre (server momentum).
        """
        delta_local_updates = kwargs["delta_local_updates"]
        tau     = kwargs.get("cclip_tau", 10.0)
        n_iter  = kwargs.get("cclip_n_iter", 1)
        n       = len(delta_local_updates)
        keys    = list(delta_local_updates[0].keys())
        device  = next(iter(delta_local_updates[0].values())).device

        # Build matrix of updates (n × d)
        with torch.no_grad():
            updates_flat = torch.stack(
                [self._flatten_update(u, keys) for u in delta_local_updates]
            ).to(device)

            # Initialise centre m
            if self.momentum is None:
                self.momentum = torch.zeros_like(updates_flat[0], device=device)

            # Fixed‑point iterations
            for _ in range(max(1, n_iter)):
                centred = updates_flat - self.momentum      # (n × d)
                clipped = torch.stack([self._clip(v, tau) for v in centred])
                self.momentum = self.momentum + clipped.mean(dim=0)

            # Convert aggregated vector back to dict
            return self._unflatten_vector(self.momentum, delta_local_updates[0], keys)

    def cclip_bucketing(self, *args, **kwargs):
        """
        Implements the s‑bucketing wrapper from Karimireddy et al.:
            • Randomly partition clients into `s` buckets
            • Run *one* round of centred clipping on each bucket (no momentum carry‑over)
            • Average the bucket outputs
        The server‑level centre `self.momentum` is still updated each global round.
        """
        delta_local_updates = kwargs["delta_local_updates"]
        s           = kwargs.get("bucketing_factor", 2)
        tau         = kwargs.get("cclip_tau", 10.0)
        n_iter      = kwargs.get("cclip_n_iter", 1)
        random.seed(kwargs.get("seed", None))

        n_clients = len(delta_local_updates)
        s = max(1, min(s, n_clients))              # guard: at least 1, at most n
        keys = list(delta_local_updates[0].keys())
        device = next(iter(delta_local_updates[0].values())).device

        # --- Random partition -------------------------------------------------
        shuffled = delta_local_updates.copy()
        random.shuffle(shuffled)
        buckets = [shuffled[i::s] for i in range(s)]

        # --- Run centred clipping *inside* each bucket ------------------------
        bucket_outputs = []
        for bucket in buckets:
            # Vectorise bucket
            bucket_mat = torch.stack(
                [self._flatten_update(u, keys) for u in bucket]
            ).to(device)
            centre = self.momentum if self.momentum is not None else torch.zeros_like(bucket_mat[0])

            # Do n_iter fixed‑point steps for this bucket (no centre carry‑over)
            m_bucket = centre.clone()
            for _ in range(max(1, n_iter)):
                centred = bucket_mat - m_bucket
                clipped = torch.stack([self._clip(v, tau) for v in centred])
                m_bucket = m_bucket + clipped.mean(dim=0)

            bucket_outputs.append(m_bucket)

        # --- Average bucket outputs & update global momentum ------------------
        self.momentum = torch.stack(bucket_outputs).mean(dim=0)

        return self._unflatten_vector(self.momentum, delta_local_updates[0], keys)

    def _bucketed_stateless(self, inner_fn, *, delta_local_updates,
                            bucketing_factor=2, seed=None, **kwargs):
        """
        Random s-bucketing wrapper for *stateless* aggregation rules.
        Works with: krum, trimmed_mean, bulyan.
        Does NOT handle stateful rules like cclip.

        Args
        ----
        inner_fn : callable(self, delta_local_updates=..., **kwargs) -> dict
        bucketing_factor : int, number of buckets s (1..n)
        seed : optional RNG seed
        kwargs : forwarded to inner_fn (with safety fixes below)
        """
        n = len(delta_local_updates)
        if n == 0:
            raise ValueError("delta_local_updates must be non-empty.")

        s = max(1, min(int(bucketing_factor), n))

        # Fast path: no bucketing
        safe_kwargs = dict(kwargs)
        safe_kwargs.pop("delta_local_updates", None)  # avoid duplicate kwarg

        # Krum guard: ensure we get only the update dict (not (dict, idx))
        if inner_fn is self.krum:
            safe_kwargs["return_index"] = False

        if s == 1:
            # Work on a copy so inner_fn (e.g., bulyan) can mutate safely
            return inner_fn(delta_local_updates=[{k: v.clone() for k, v in u.items()}
                                                 for u in delta_local_updates],
                            **safe_kwargs)

        # Random partition
        random.seed(seed)
        shuffled = delta_local_updates.copy()
        random.shuffle(shuffled)
        buckets = [shuffled[i::s] for i in range(s)]

        bucket_outs = []
        for bucket in buckets:
            # Deep(ish) copy: clone tensors, preserve structure
            bucket_copy = [{k: v.clone() for k, v in u.items()} for u in bucket]
            out = inner_fn(delta_local_updates=bucket_copy, **safe_kwargs)
            bucket_outs.append(out)

        # Average bucket outputs (dict-of-tensors)
        keys = bucket_outs[0].keys()
        aggregated = {
            k: torch.stack([bo[k] for bo in bucket_outs]).mean(dim=0)
            for k in keys
        }
        return aggregated

    def bulyan_bucketing(self, *args, **kwargs):
        """Apply bulyan defence with bucketing with adjusted factors."""
        n = len(kwargs['delta_local_updates'])
        s = kwargs.get('bucketing_factor', 2)
        
        # Calculate f from krum factor first
        original_krum_factor = kwargs.get('krum_factor', n - 2)
        f = n - original_krum_factor - 2
        
        # Set bulyan factor as min(f, n//4) or use provided value
        original_bulyan_factor = kwargs.get('bulyan_factor', min(f, n // 4))

        # Adjust bulyan factor for bucket size
        bucket_size = n // s
        adjusted_bulyan_factor = max(1, original_bulyan_factor // s)
        
        # Adjust f for bucket size
        adjusted_f = max(1, f // s)
        adjusted_krum_factor = bucket_size - adjusted_f - 2
        
        # Create new kwargs with adjusted factors
        new_kwargs = dict(kwargs)
        new_kwargs['bulyan_factor'] = adjusted_bulyan_factor
        new_kwargs['krum_factor'] = adjusted_krum_factor
        
        return self._bucketed_stateless(self.bulyan, *args, **new_kwargs)

    def rfa(self, *args, **kwargs):
        """
        Robust Federated Aggregation (RFA) via smoothed Weiszfeld algorithm.
        Implements geometric median computation.
        
        Args:
            delta_local_updates (list[dict]): List of client updates
            client_weights (list[float], optional): Client weights α_i
            rfa_nu (float, optional): Smoothing parameter ν > 0. Default: 1e-6
            rfa_R (int, optional): Number of Weiszfeld iterations. Default: 3
            rfa_init (str, optional): Initialization method {'mean','zero','first'}. Default: 'mean'
        
        Returns:
            dict: Aggregated update (approximate geometric median)
        """
        delta_local_updates = kwargs["delta_local_updates"]
        n = len(delta_local_updates)
        if n == 0:
            raise ValueError("delta_local_updates must be non-empty")
        
        keys = list(delta_local_updates[0].keys())
        device = next(iter(delta_local_updates[0].values())).device
        dtype = next(iter(delta_local_updates[0].values())).dtype

        # Stack updates as n × d matrix
        t = time.time()
        with torch.no_grad():
            W = torch.stack(
                [self._flatten_update(u, keys) for u in delta_local_updates],
                dim=0
            ).to(device=device, dtype=dtype)  # (n,d)

            # Client weights α_i
            alpha = kwargs.get("client_weights", None)
            if alpha is None:
                alpha = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
            else:
                alpha = torch.as_tensor(alpha, device=device, dtype=dtype)
                if alpha.numel() != n:
                    raise ValueError("client_weights length must match number of clients")
                # Normalize to sum 1
                alpha = alpha / alpha.sum()

            # Algorithm parameters
            nu = float(kwargs.get("rfa_nu", 1e-6))  # Smoothing parameter
            R = int(kwargs.get("rfa_R", 3))         # Number of iterations
            init = kwargs.get("rfa_init", "mean")    # Initialization method

            # Initialize v^(0)
            if init == "zero":
                v = torch.zeros(W.shape[1], device=device, dtype=dtype)
            elif init == "first":
                v = W[0].clone()
            else:  # 'mean'
                v = (alpha.view(-1, 1) * W).sum(dim=0)

            # Weiszfeld iterations
            for _ in range(max(1, R)):
                # Compute distances, clamped by ν
                dist = torch.norm(W - v, dim=1).clamp_min(nu)  # (n,)
                beta = alpha / dist  # β_i = α_i / max(ν, ||v - w_i||)
                denom = beta.sum()
                
                # Guard against numerical issues
                if denom.item() == 0.0 or not torch.isfinite(denom):
                    # Fallback to weighted mean
                    v = (alpha.view(-1, 1) * W).sum(dim=0)
                    break
                    
                v = (beta.view(-1, 1) * W).sum(dim=0) / denom

            # Rescale v to have norm similar to weighted mean
            m = (alpha.view(-1, 1) * W).sum(dim=0)          # weighted mean
            nm = torch.norm(m)
            nv = torch.norm(v)
            s  = (nm / (nv + 1e-12)).clamp(0.25, 4.0)      # optional safety clamp
            v  = v * s

            print(f"RFA time: {time.time() - t:.3f}s")
            return self._unflatten_vector(v, delta_local_updates[0], keys)

    def rfa_bucketing(self, *args, **kwargs):
        """
        Apply RFA with s-bucketing (stateless wrapper).
        
        Args:
            bucketing_factor (int): Number of buckets s (default: 2)
            seed (optional): Random seed for bucket assignment
            Other kwargs are passed to inner RFA
        
        Returns:
            dict: Aggregated update from averaged bucket medians
        """
        return self._bucketed_stateless(self.rfa, *args, **kwargs)

    @torch.no_grad()
    def huber(self,
            *,
            delta_local_updates: list[dict[str, torch.Tensor]],
            byzantine_frac: float | None = 0.0,
            huber_delta: float | None = None,
            huber_tau: float | list[float] | None = None,
            sample_sizes: list[int] | None = None,
            tau_rule: str = "balanced",
            hetero_offset: float = 0.0,
            weight_by_samples: bool = False,
            samples_per_client: int | None = None,
            total_samples: int | None = None,
            huber_max_iter: int = 50,
            huber_tol: float = 1e-6,
            **_) -> dict[str, torch.Tensor]:
        updates = delta_local_updates
        m = len(updates)
        if m == 0:
            raise ValueError("delta_local_updates must be non-empty")
        if sample_sizes is not None and len(sample_sizes) != m:
            raise ValueError("sample_sizes must have length m")

        keys   = list(updates[0].keys())
        device = next(iter(updates[0].values())).device
        dtype  = next(iter(updates[0].values())).dtype

        # ---- tau builder (same logic as your vectorized version), O(1) memory ----
        def build_tau_vec() -> torch.Tensor:
            # dimension d without building X
            d = sum(p.numel() for p in updates[0].values())
            _delta = huber_delta
            if _delta is None:
                _delta = min(0.1, max(1e-10, 1.0 / (m * m)))

            # explicit τ overrides
            if huber_tau is not None:
                if isinstance(huber_tau, (list, tuple, torch.Tensor)):
                    tv = torch.as_tensor(huber_tau, dtype=dtype, device=device)
                    if tv.numel() != m:
                        raise ValueError("huber_tau list must match #clients")
                    return tv + torch.tensor(float(hetero_offset), dtype=dtype, device=device)
                return torch.full((m,), float(huber_tau) + float(hetero_offset), dtype=dtype, device=device)

            # choose n_i and N
            if sample_sizes is not None:
                n_vec = torch.as_tensor(sample_sizes, dtype=torch.float32, device=device).clamp_min_(1.0)
                N = float(n_vec.sum().item())
            elif samples_per_client is not None:
                val = max(1.0, float(samples_per_client))
                n_vec = torch.full((m,), val, dtype=torch.float32, device=device)
                N = float(val * m)
            elif total_samples is not None:
                N = float(max(1, int(total_samples)))
                n_val = max(1.0, N / m)
                n_vec = torch.full((m,), n_val, dtype=torch.float32, device=device)
            else:
                n_vec = None
                N = float(m)

            log_term = math.log(max(N, 2.0) / max(_delta, 1e-12))

            if tau_rule == "unbalanced_simple" and n_vec is not None:
                tv = 2.0 / torch.sqrt(n_vec)
                return tv.to(dtype) + torch.tensor(float(hetero_offset), dtype=dtype, device=device)

            if tau_rule in ("unbalanced_theory", "balanced_noniid") and n_vec is not None:
                M   = math.sqrt(max(d, 1) * log_term)
                eps = float(byzantine_frac) if (byzantine_frac is not None) else 0.0
                eps = max(0.0, min(0.49, eps))
                T0  = (eps / max(1e-6, 1.0 - 2.0 * eps)) * M * math.sqrt(max(m, 1) / max(N, 1e-6))
                tv  = torch.tensor(T0, dtype=dtype, device=device) + torch.tensor(M, dtype=dtype, device=device) / torch.sqrt(n_vec.to(dtype))
                return tv + torch.tensor(float(hetero_offset), dtype=dtype, device=device)

            # balanced / fallback
            if n_vec is not None and N > 0:
                n_proxy = max(1.0, N / m)
            else:
                n_proxy = max(1.0, m)
            base = math.sqrt(max(d, 1) / n_proxy) * math.sqrt(log_term)
            return torch.full((m,), base + float(hetero_offset), dtype=dtype, device=device)

        tau_vec = build_tau_vec().clamp_min_(1e-12).to(device=device, dtype=dtype)

        # ---- init centre at simple mean (dict-of-tensors), O(d) memory ----------
        centre = {k: torch.zeros_like(v, device=device, dtype=dtype) for k, v in updates[0].items()}
        for u in updates:
            for k in keys: centre[k] += u[k]
        for k in keys: centre[k] /= m
        mean_fallback = {k: t.clone() for k, t in centre.items()}  # for rare denom=0 fallback

        one = torch.tensor(1.0, device=device, dtype=torch.float32)
        eps = torch.tensor(1e-12, device=device, dtype=torch.float32)
        samp_w = (torch.as_tensor(sample_sizes, dtype=torch.float32, device=device).clamp_min_(1.0)
                if (weight_by_samples and sample_sizes is not None) else None)

        for _ in range(max(1, int(huber_max_iter))):
            num_agg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in centre.items()}
            den_agg = torch.tensor(0.0, dtype=torch.float32, device=device)

            for i, u in enumerate(updates):
                # dist_i = ||u_i - centre||_2
                dist2 = torch.tensor(0.0, dtype=torch.float32, device=device)
                for k in keys:
                    diff = (u[k].float() - centre[k].float())
                    dist2 += (diff * diff).sum()
                dist = torch.sqrt(dist2 + eps)

                w = torch.minimum(tau_vec[i].float() / torch.maximum(dist, eps), one)
                if samp_w is not None:
                    w = w * samp_w[i]

                den_agg += w
                for k in keys:
                    num_agg[k] += w * u[k].float()

            if not torch.isfinite(den_agg) or den_agg.item() <= 0.0:
                new_c = mean_fallback  # true mean, not num_agg/m (which was weighted)
            else:
                new_c = {k: (v / den_agg).to(dtype) for k, v in num_agg.items()}

            # relative stop
            nume = torch.tensor(0.0, dtype=torch.float32, device=device)
            deno = torch.tensor(0.0, dtype=torch.float32, device=device)
            for k in keys:
                dk = (new_c[k].float() - centre[k].float())
                nume += (dk * dk).sum()
                deno += (centre[k].float() * centre[k].float()).sum()

            if torch.sqrt(nume) <= huber_tol * torch.sqrt(torch.maximum(deno, torch.tensor(1.0, device=device))):
                centre = new_c
                break
            centre = new_c

        return centre

    @torch.no_grad()
    def coord_median(self, *, delta_local_updates, clip_norm=None, **kwargs):
        """Coordinate-wise median aggregator (Yin et al., 2018).

        Optionally applies per-client ℓ2 norm clipping before taking medians.

        Args
        ----
        delta_local_updates : list[dict[str, Tensor]]
            Per-client updates with identical keys/shapes.
        clip_norm : float | None
            If provided, each client's *full* update vector is scaled so ‖u_i‖₂ ≤ clip_norm.

        Returns
        -------
        dict[str, Tensor] – aggregated update
        """
        if len(delta_local_updates) == 0:
            raise ValueError("delta_local_updates must be non-empty")

        keys   = list(delta_local_updates[0].keys())
        device = next(iter(delta_local_updates[0].values())).device
        dtype  = next(iter(delta_local_updates[0].values())).dtype
        m      = len(delta_local_updates)

        # Precompute per-client clipping scales once from the *flattened* vectors
        if clip_norm is not None:
            flat_mat = torch.stack(
                [self._flatten_update(u, keys) for u in delta_local_updates],
                dim=0
            ).to(device=device, dtype=dtype)  # (m, d)
            norms = flat_mat.norm(p=2, dim=1) + 1e-12  # (m,)
            scales = torch.clamp(clip_norm / norms, max=1.0)  # (m,)
        else:
            scales = None

        aggregated = {}
        for k in keys:
            stacked = torch.stack([u[k].to(device=device, dtype=dtype) for u in delta_local_updates], dim=0)
            if scales is not None:
                view_shape = (m,) + (1,) * (stacked.dim() - 1)
                stacked = stacked * scales.view(view_shape)
            aggregated[k] = stacked.median(dim=0).values

        return aggregated

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)