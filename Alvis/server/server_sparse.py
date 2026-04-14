import copy
import torch
import wandb
import numpy as np

from server.server_base import BaseServer


class SparseFLServer(BaseServer):
    """
    Implements a Sparse Federated Learning strategy using line searches on alpha and beta,
    projection onto a simplex, and optional partial/total loss weighting.
    """

    def run(
        self,
        alpha,
        beta,
        is_ftotal=True,
        k_value=(0, None, None, None, None),
        c_alpha=1e-4,
        rho_alpha=0.5,
        max_line_search_iterations_alpha=0,
        c_beta=1e-2,
        rho_beta=0.5,
        max_line_search_iterations_beta=10,
        estimate_initial_updates=False,
        normalize_gradients=False,
    ):
        num_clients = len(self.clients)

        # ===============================================================
        # INITIAL ROUND — no attack, collect clean gradients + losses
        # ===============================================================
        # ✅ FIX: clone so optimizer.step() on clients doesn't mutate this
        global_weights = {
            k: v.detach().clone()
            for k, v in self.global_model.state_dict().items()
        }

        G, F_T_next, clean_ids = self._gather_client_updates(
            global_weights,
            epoch=0,
            lr=self.learning_rate,   # ✅ FIXED: use stored lr, not alpha
            return_avg_loss=True,
            compute_gradient=True,
            apply_attack=False,
        )

        # ✅ Guard: if too many clients dropped in round 0, something is
        #    fundamentally broken (likely float16 NaN from Hadamard transform)
        if len(G) < self.num_clients * 0.5:
            raise RuntimeError(
                f"Too many clients dropped in initial round: "
                f"only {len(G)}/{self.num_clients} survived. "
                f"Check for NaN losses — likely float16 overflow in Hadamard transform. "
                f"Ensure fast_hadamard_transform casts to float32 internally."
            )

        active_ids = clean_ids

        # Uniform initialisation over surviving clients
        w = [1.0 / len(G)] * len(G)

        total_epochs = self.total_epochs
        k_values = [len(G)] * total_epochs

        # Upper bound per client weight for capped-simplex projection
        maximum_weight_bound = 1.0 / int((1 - self.fraction_malicious) * num_clients)

        # ===============================================================
        # MAIN TRAINING LOOP
        # ===============================================================
        for epoch in range(self.total_epochs):

            print(f"\n===== Round {epoch + 1}/{self.total_epochs} =====")
            print(f"[INFO] Active clients : {len(G)}")
            print(f"[INFO] Alpha (fed step): {alpha:.6f}")
            print(f"[INFO] Client lr      : {self.learning_rate}")

            # -----------------------------------------------------------
            # 1) LINE SEARCH ON ALPHA (no attack inside)
            # -----------------------------------------------------------
            alpha = self._line_search_alpha(
                alpha=alpha,
                G=G,
                F_T_next=F_T_next,
                w=w,
                c=c_alpha,
                rho=rho_alpha,
                epoch=epoch,
                max_iteration=max_line_search_iterations_alpha,
            )

            # -----------------------------------------------------------
            # 2) FIRST THETA UPDATE (pre-attack, uses previous G)
            # -----------------------------------------------------------
            self._theta_update(
                G=G,
                G_next=None,
                F_T_next=F_T_next,
                w=w,
                alpha=alpha,
                epoch=epoch,
                compute_gradient=False,
            )

            # -----------------------------------------------------------
            # 3) GATHER NEW CLIENT UPDATES — attack applied HERE ONLY
            # -----------------------------------------------------------
            client_gradients, client_losses, clean_ids = self._gather_client_updates(
                {k: v.detach().clone() for k, v in self.global_model.state_dict().items()},
                epoch=epoch,
                lr=self.learning_rate,   # ✅ FIXED: use stored lr, not alpha
                return_avg_loss=True,
                compute_gradient=True,
                apply_attack=True,
            )

            # -----------------------------------------------------------
            # 4) ALIGN CLIENT IDS (handles dropouts between rounds)
            # -----------------------------------------------------------
            id_map = {cid: i for i, cid in enumerate(active_ids)}

            new_w = []
            new_G = []
            new_F = []

            for i, cid in enumerate(clean_ids):
                if cid in id_map:
                    new_w.append(w[id_map[cid]])
                    new_G.append(client_gradients[i])
                    new_F.append(client_losses[i])

            if len(new_w) == 0:
                raise RuntimeError(
                    "All clients lost during ID alignment — "
                    "check that client IDs are stable across rounds."
                )

            w          = new_w
            G          = new_G
            F_T_next   = new_F
            active_ids = clean_ids

            assert len(w) == len(G),        f"Weight/gradient mismatch: w={len(w)}, G={len(G)}"
            assert len(F_T_next) == len(G), f"Loss/gradient mismatch: F={len(F_T_next)}, G={len(G)}"

            avg_loss_before = float(np.dot(F_T_next, w))
            print(f"[Round {epoch+1}] Weighted loss before weight update: {avg_loss_before:.4f}")
            print(f"[Round {epoch+1}] Weights before update: {[round(wi, 4) for wi in w]}")

            # -----------------------------------------------------------
            # 5) FEDLAW WEIGHT UPDATE
            #    G     = gradients at theta_t   (before this round's update)
            #    G_next = gradients at theta_t+1 (after second theta update)
            #    The cross-round dot product G^T G_next is what detects
            #    Byzantine clients — malicious gradients at t should have
            #    low alignment with the honest direction at t+1.
            # -----------------------------------------------------------

            # Second theta update first — needed to get G_next
            self._theta_update(
                G=G,
                G_next=None,
                F_T_next=F_T_next,
                w=w,
                alpha=alpha,
                epoch=epoch,
                compute_gradient=False,
            )

            # Gather G_next at updated theta (no attack — clean signal)
            G_next, _, _ = self._gather_client_updates(
                {k: v.detach().clone() for k, v in self.global_model.state_dict().items()},
                epoch=epoch,
                lr=self.learning_rate,
                return_avg_loss=True,
                compute_gradient=True,
                apply_attack=False,   # ✅ G_next must be clean — no attack
            )

            # Align G_next with current active client order
            new_G_next = []
            for cid in active_ids:
                # find index of this client in the G_next gather
                # active_ids already aligned after step 4, so positional match
                new_G_next.append(G_next[active_ids.index(cid)] if cid in active_ids else G[active_ids.index(cid)])
            G_next = new_G_next[:len(G)]

            w = self._weight_update(
                G=G,
                G_next=G_next,        # ✅ FIXED: cross-round gradients
                F_T_next=F_T_next,
                w=w,
                alpha=alpha,
                beta=beta,
                k_value=min(k_values[epoch], len(G)),
                maximum_weight_bound=maximum_weight_bound,
                is_ftotal=is_ftotal,
                max_line_search_iterations=max_line_search_iterations_beta,
                c_beta=c_beta,
                rho_beta=rho_beta,
                normalize_gradients=normalize_gradients,
            )

            print(f"[Round {epoch+1}] Weights after update : {[round(wi, 4) for wi in w]}")

            avg_loss_after = float(np.dot(F_T_next, w))
            print(f"[Round {epoch+1}] Weighted loss after  weight update: {avg_loss_after:.4f}")

            # -----------------------------------------------------------
            # 7) EVALUATION
            # -----------------------------------------------------------
            if epoch % self.evaluate_each_epoch == 0:
                test_acc, test_loss = self.calculate_accuracy(is_fedavg=False)
                wandb.log({
                    "epoch":        epoch + 1,
                    "test_accuracy": test_acc,
                    "test_loss":    test_loss,
                    "alpha":        alpha,
                    "avg_loss_before_weight_update": avg_loss_before,
                    "avg_loss_after_weight_update":  avg_loss_after,
                })

    # ===================================================================
    # LINE SEARCH: ALPHA (federated step size)
    # ===================================================================
    def _line_search_alpha(self, alpha, G, F_T_next, w, c, rho, epoch, max_iteration=3):

        if max_iteration == 0:
            return alpha

        params_new = {k: v.clone() for k, v in self.global_model.state_dict().items()}

        for _ in range(max_iteration):

            agg_grad_vector = []

            with torch.no_grad():
                for name, param in self.global_model.named_parameters():

                    if 'lora_' not in name:
                        continue

                    agg_grad = torch.zeros_like(param)

                    for client_idx, grad_dict in enumerate(G):
                        client_grad = grad_dict[name].to(param.device)

                        if client_grad.shape != param.shape:
                            if client_grad.T.shape == param.shape:
                                client_grad = client_grad.T
                            elif client_grad.numel() == param.numel():
                                client_grad = client_grad.reshape(param.shape)
                            else:
                                raise RuntimeError(
                                    f"Shape mismatch in alpha line search for {name}: "
                                    f"{client_grad.shape} vs {param.shape}"
                                )

                        agg_grad += float(w[client_idx]) * client_grad

                    params_new[name] = param - alpha * agg_grad
                    agg_grad_vector.append(agg_grad.view(-1))

            # Evaluate loss at proposed new params — no attack, no gradient needed
            new_client_losses = self._gather_client_updates(
                params_new,
                epoch=epoch,
                lr=self.learning_rate,   # ✅ FIXED: use stored lr, not alpha
                return_avg_loss=True,
                compute_gradient=False,
                apply_attack=False,
            )[1]

            L_new = np.mean(new_client_losses)
            L_old = np.mean(F_T_next)

            agg_grad_tensor = torch.cat(agg_grad_vector)
            sufficient_decrease = c * alpha * torch.norm(agg_grad_tensor).item() ** 2

            if L_new <= L_old - sufficient_decrease:
                print(f"[LineSearch alpha] Accepted alpha={alpha:.6f} | L_new={L_new:.4f} <= L_old={L_old:.4f} - {sufficient_decrease:.6f}")
                return alpha

            alpha *= rho
            print(f"[LineSearch alpha] Reducing alpha → {alpha:.6f}")

        print("[LineSearch alpha] Did not converge — using final alpha")
        return alpha

    # ===================================================================
    # THETA UPDATE (global model parameter update)
    # ===================================================================
    def _theta_update(self, G, G_next, F_T_next, w, alpha, epoch, params_copy=None, compute_gradient=True):

        alpha = float(alpha)

        with torch.no_grad():
            for name, param in self.global_model.named_parameters():

                if 'lora_' not in name:
                    continue

                agg_grad = torch.zeros_like(param)

                for client_idx, grad_dict in enumerate(G):

                    if name not in grad_dict:
                        continue

                    client_grad = grad_dict[name].to(param.device)

                    if client_grad.shape != param.shape:
                        if client_grad.T.shape == param.shape:
                            client_grad = client_grad.T
                        elif client_grad.numel() == param.numel():
                            client_grad = client_grad.reshape(param.shape)
                        else:
                            raise RuntimeError(
                                f"Shape mismatch in theta update for {name}: "
                                f"{client_grad.shape} vs {param.shape}"
                            )

                    agg_grad += float(w[client_idx]) * client_grad

                if params_copy is None:
                    param -= alpha * agg_grad
                else:
                    if name not in params_copy:
                        raise ValueError(f"Parameter {name} not found in params_copy.")
                    param.copy_(params_copy[name] - alpha * agg_grad)

        return None, None

    # ===================================================================
    # WEIGHT UPDATE (FedLAW)
    # ===================================================================
    def _weight_update(
        self,
        G,
        G_next,
        F_T_next,
        w,
        alpha,
        beta,
        k_value,
        maximum_weight_bound,
        is_ftotal,
        max_line_search_iterations,
        c_beta,
        rho_beta,
        eye_factor=1e-6,
        normalize_gradients=False,
    ):
        device = self.device
        alpha  = float(alpha)
        beta   = float(beta)

        assert len(w) == len(G),        f"w={len(w)}, G={len(G)}"
        assert len(F_T_next) == len(G), f"F={len(F_T_next)}, G={len(G)}"

        w_tensor          = torch.tensor(w,          dtype=torch.float32, device=device)
        F_T_next_tensor   = torch.tensor(F_T_next,   dtype=torch.float32, device=device)

        # -----------------------------------------------------------
        # FLATTEN gradients → columns matrix (d x n)
        # -----------------------------------------------------------
        G_flat      = self._flatten_tensors(G,      normalize=False).to(device)
        G_next_flat = self._flatten_tensors(G_next, normalize=False).to(device)

        # -----------------------------------------------------------
        # FEDLAW CORE: G^T G_next  (n x n)
        # -----------------------------------------------------------
        G_T_G_next  = torch.matmul(G_flat.T, G_next_flat)
        G_T_G_next += eye_factor * torch.eye(G_T_G_next.shape[0], device=device)

        G_T_G_next_w = torch.matmul(G_T_G_next, w_tensor)

        print("\n---- WEIGHT DEBUG ----")
        print(f"  grad term norm : {torch.norm(G_T_G_next_w).item():.6f}")
        print(f"  loss term norm : {torch.norm(F_T_next_tensor).item():.6f}")

        # -----------------------------------------------------------
        # COMPUTE m_next
        # -----------------------------------------------------------
        if is_ftotal is True or is_ftotal == "fedlaw":
            m_next = w_tensor + alpha * beta * G_T_G_next_w - beta * F_T_next_tensor
        elif is_ftotal == "bsum":
            m_next = w_tensor - beta * F_T_next_tensor
        else:
            m_next = w_tensor + alpha * beta * G_T_G_next_w

        # -----------------------------------------------------------
        # PROJECT onto capped simplex
        # -----------------------------------------------------------
        w_next = self._sparse_projection_onto_simplex(
            m_next=m_next.detach().cpu().numpy(),
            k_value=k_value,
            t=maximum_weight_bound,
        )

        # -----------------------------------------------------------
        # LINE SEARCH ON BETA
        # -----------------------------------------------------------
        beta, w_next, m_next = self._line_search_for_beta(
            w_tensor, m_next, w_next, alpha, beta,
            G_T_G_next_w, F_T_next_tensor, is_ftotal,
            k_value, maximum_weight_bound,
            max_line_search_iterations, c_beta, rho_beta,
        )

        # -----------------------------------------------------------
        # Pure FedLAW output — Byzantine robustness comes entirely
        # from the G^T G_next alignment term and the -β·F loss term.
        # No external norm penalty is needed or applied.
        # -----------------------------------------------------------
        print("\n====== WEIGHT UPDATE DEBUG ======")
        print(f"  Final weights  : {[round(wi, 4) for wi in w_next]}")

        return w_next

    # ===================================================================
    # LINE SEARCH: BETA
    # ===================================================================
    def _line_search_for_beta(
        self,
        w_tensor,
        m_next,
        w_next_normalize,
        alpha,
        beta,
        G_T_G_next_w,
        F_T_next_tensor,
        is_ftotal,
        k_value,
        maximum_weight_bound,
        max_line_search_iterations,
        c_beta,
        rho_beta,
    ):
        if max_line_search_iterations == 0:
            return beta, w_next_normalize, m_next

        for _ in range(max_line_search_iterations):

            w_next_tensor = torch.tensor(
                w_next_normalize,
                dtype=torch.float32,
                device=self.device,
            )

            f_new     = 0.5 * torch.norm(w_next_tensor - m_next) ** 2
            f_current = 0.5 * torch.norm(w_tensor     - m_next) ** 2
            grad_f    = torch.abs((w_tensor - m_next).dot(w_next_tensor - w_tensor))

            if f_new <= f_current - c_beta * beta * grad_f:
                break

            beta *= rho_beta

            if is_ftotal:
                m_next = w_tensor + alpha * beta * G_T_G_next_w - beta * F_T_next_tensor
            else:
                m_next = w_tensor + alpha * beta * G_T_G_next_w

            w_next_normalize = self._sparse_projection_onto_simplex(
                m_next=m_next.detach().cpu().numpy(),
                k_value=k_value,
                t=maximum_weight_bound,
            )

        return beta, w_next_normalize, m_next

    # ===================================================================
    # SIMPLEX PROJECTION (dispatcher)
    # ===================================================================
    def _sparse_projection_onto_simplex(self, *args, **kwargs):
        t_val = kwargs.get("t", None)
        if t_val is not None:
            return self._sparse_projection_capped_simplex(*args, **kwargs)
        else:
            kwargs.pop("t", None)
            return self._sparse_projection_onto_unit_simplex(*args, **kwargs)

    # ===================================================================
    # UNIT SIMPLEX PROJECTION
    # ===================================================================
    def _sparse_projection_onto_unit_simplex(self, m_next, k_value):
        m_next = np.array(m_next, dtype=float)

        if k_value < 1 or k_value > len(m_next):
            return [0.0] * len(m_next)

        idxs_desc  = np.argsort(m_next)[::-1]
        sorted_m   = m_next[idxs_desc]
        top_k_idxs = idxs_desc[:k_value]
        P_L_lambda = sorted_m[:k_value]

        cumsum_vals = np.cumsum(P_L_lambda)
        rhos        = P_L_lambda > (cumsum_vals - 1.0) / np.arange(1, k_value + 1)

        if np.any(rhos):
            rho_idx = np.where(rhos)[0].max()
            eta     = (cumsum_vals[rho_idx] - 1.0) / (rho_idx + 1.0)
        else:
            eta = cumsum_vals[-1] / k_value

        P_plus  = np.maximum(P_L_lambda - eta, 0)
        w_proj  = np.zeros_like(m_next)
        w_proj[top_k_idxs] = P_plus

        return w_proj.tolist()

    # ===================================================================
    # CAPPED SIMPLEX PROJECTION
    # ===================================================================
    def _sparse_projection_capped_simplex(self, m_next, k_value, t):
        m_next = np.array(m_next, dtype=float)
        n_full = len(m_next)
        k      = 1.0

        if k_value < 1:
            raise ValueError("k_value < 1 is invalid when sum must equal 1.")
        if k_value * t < k:
            raise ValueError(f"Infeasible: k_value * t = {k_value * t} < 1.")

        idxs_desc  = np.argsort(m_next)[::-1]
        k_value    = min(k_value, n_full)
        valid_mask = idxs_desc[:k_value]
        y0         = m_next[valid_mask]

        y0_scaled  = y0 / t
        k_scaled   = k  / t
        x_part     = np.zeros(k_value, dtype=float)

        idx_asc = np.argsort(y0_scaled)
        y_asc   = y0_scaled[idx_asc]
        s_cum   = np.cumsum(y_asc)
        y_asc   = np.append(y_asc, np.inf)

        for b in range(1, k_value + 1):
            gamma = (k_scaled + b - k_value - s_cum[b - 1]) / b
            if (y_asc[0] + gamma > 0) and (y_asc[b - 1] + gamma < 1) and (y_asc[b] + gamma >= 1):
                xtmp = np.concatenate((y_asc[:b] + gamma, np.ones(k_value - b)))
                xtmp *= t
                x_part[idx_asc] = xtmp
                w_proj          = np.zeros(n_full, dtype=float)
                w_proj[valid_mask] = x_part
                if np.isclose(np.sum(w_proj), 1.0, atol=1e-7):
                    return w_proj.tolist()

        for a in range(1, k_value):
            for b in range(a + 1, k_value + 1):
                gamma = (k_scaled + b - k_value + s_cum[a - 1] - s_cum[b - 1]) / (b - a)
                if (
                    (y_asc[a - 1] + gamma <= 0) and
                    (y_asc[a]     + gamma >  0) and
                    (y_asc[b - 1] + gamma <  1) and
                    (y_asc[b]     + gamma >= 1)
                ):
                    xtmp = np.concatenate((np.zeros(a), y_asc[a:b] + gamma, np.ones(k_value - b)))
                    xtmp *= t
                    x_part[idx_asc] = xtmp
                    w_proj          = np.zeros(n_full, dtype=float)
                    w_proj[valid_mask] = x_part
                    if np.isclose(np.sum(w_proj), 1.0, atol=1e-7):
                        return w_proj.tolist()

        # Fallback: greedily fill up to t
        needed  = 1.0
        x_part  = np.zeros(k_value, dtype=float)
        for i in range(k_value):
            if needed >= t:
                x_part[i] = t
                needed    -= t
            else:
                x_part[i] = needed
                needed     = 0.0
                break

        w_proj = np.zeros(n_full, dtype=float)
        w_proj[valid_mask] = x_part

        if not np.isclose(np.sum(w_proj), 1.0, atol=1e-7):
            raise ValueError(
                f"Capped simplex projection failed: sum={np.sum(w_proj):.6f}, "
                f"expected 1.0. k_value={k_value}, t={t}"
            )

        return w_proj.tolist()

    # ===================================================================
    # FLATTEN TENSORS — sparse server version (supports normalize flag)
    # ===================================================================
    def _flatten_tensors(self, input_list, normalize=False):

        if normalize:
            norms = []
            for client_grads in input_list:
                squared_norm = sum(
                    torch.sum(grad * grad)
                    for grad in client_grads.values()
                )
                norms.append(torch.sqrt(squared_norm))

            median_norm = torch.median(torch.stack(norms))

            flattened = []
            for i, client_grads in enumerate(input_list):
                scale = median_norm / norms[i] if norms[i] > 0 else 1.0
                flat  = torch.cat([
                    (client_grads[key] * scale).view(-1)
                    for key in sorted(client_grads.keys())
                ])
                flattened.append(flat)

        else:
            flattened = [
                torch.cat([
                    client_grads[key].view(-1)
                    for key in sorted(client_grads.keys())
                ])
                for client_grads in input_list
            ]

        return torch.stack(flattened, dim=1)