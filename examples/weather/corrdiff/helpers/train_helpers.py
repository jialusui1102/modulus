# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import torch
import numpy as np
from omegaconf import ListConfig
import torch.distributed as dist
import wandb
import math
import copy
import matplotlib.pyplot as plt

def set_patch_shape(img_shape, patch_shape):
    img_shape_y, img_shape_x = img_shape
    patch_shape_y, patch_shape_x = patch_shape
    if (patch_shape_x is None) or (patch_shape_x > img_shape_x):
        patch_shape_x = img_shape_x
    if (patch_shape_y is None) or (patch_shape_y > img_shape_y):
        patch_shape_y = img_shape_y
    if patch_shape_x == img_shape_x and patch_shape_y == img_shape_y:
        use_patching = False
    else:
        use_patching = True
    if use_patching:
        if patch_shape_x != patch_shape_y:
            warnings.warn(
                f"You are using rectangular patches "
                f"of shape {(patch_shape_y, patch_shape_x)}, "
                f"which are an experimental feature."
            )
            raise NotImplementedError("Rectangular patch not supported yet")
        if patch_shape_x % 32 != 0 or patch_shape_y % 32 != 0:
            raise ValueError("Patch shape needs to be a multiple of 32")
    return use_patching, (img_shape_y, img_shape_x), (patch_shape_y, patch_shape_x)


def set_seed(rank):
    """
    Set seeds for NumPy and PyTorch to ensure reproducibility in distributed settings
    """
    np.random.seed(rank % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))


def configure_cuda_for_consistent_precision():
    """
    Configures CUDA and cuDNN settings to ensure consistent precision by
    disabling TensorFloat-32 (TF32) and reduced precision settings.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def compute_num_accumulation_rounds(total_batch_size, batch_size_per_gpu, world_size):
    """
    Calculate the total batch size per GPU in a distributed setting, log the batch size per GPU, ensure it's within valid limits,
    determine the number of accumulation rounds, and validate that the global batch size matches the expected value.
    """
    batch_gpu_total = total_batch_size // world_size
    batch_size_per_gpu = batch_size_per_gpu
    if batch_size_per_gpu is None or batch_size_per_gpu > batch_gpu_total:
        batch_size_per_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_size_per_gpu
    if total_batch_size != batch_size_per_gpu * num_accumulation_rounds * world_size:
        raise ValueError(
            "total_batch_size must be equal to batch_size_per_gpu * num_accumulation_rounds * world_size"
        )
    return batch_gpu_total, num_accumulation_rounds


def handle_and_clip_gradients(model, grad_clip_threshold=None):
    """
    Handles NaNs and infinities in the gradients and optionally clips the gradients.

    Parameters:
    - model (torch.nn.Module): The model whose gradients need to be processed.
    - grad_clip_threshold (float, optional): The threshold for gradient clipping. If None, no clipping is performed.
    """
    # Replace NaNs and infinities in gradients
    for param in model.parameters():
        if param.grad is not None:
            torch.nan_to_num(
                param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad
            )

    # Clip gradients if a threshold is provided
    if grad_clip_threshold is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_threshold)


def parse_model_args(args):
    """Convert ListConfig values in args to tuples."""
    return {k: tuple(v) if isinstance(v, ListConfig) else v for k, v in args.items()}


def is_time_for_periodic_task(
    cur_nimg, freq, done, batch_size, rank, rank_0_only=False
):
    """Should we perform a task that is done every `freq` samples?"""
    if rank_0_only and rank != 0:
        return False
    elif done:  # Run periodic tasks also at the end of training
        return True
    else:
        return cur_nimg % freq < batch_size
    
class GradNormEMA:
    def __init__(self, decay=0.99, eps=1e-8):
        self.ema = None
        self.decay = decay
        self.eps = eps

    def update(self, grad_norm):
        if self.ema is None:
            self.ema = grad_norm
        else:
            self.ema = self.decay * self.ema + (1 - self.decay) * grad_norm
        return self.ema

    def get_threshold(self, scale=2.0):
        return scale * (self.ema + self.eps)

class SigmaLossEmaUpdater:
    def __init__(
        self,
        device,
        rank,
        sigma_bins_params = None,
        log_to_wandb = False,
        is_distributed = False,
        
        
    ):
        self.device = device
        self.rank = rank
        self.log_to_wandb = log_to_wandb
        self.is_distributed = is_distributed
        self.sigma_num_bins = sigma_bins_params.get("num_bins", 100)
        self.sigma_ema_decay = sigma_bins_params.get("ema_decay", 0.95) #0.999 train a few more with new ema decay
        self.sigma_update_every = sigma_bins_params.get("update_every", 1)
        self.sigma_eps = sigma_bins_params.get("eps", 1e-8)
        self.sigma_warmup_iterations = sigma_bins_params.get(
            "warmup_iterations", 100
        )
        self.sigma_min = sigma_bins_params.get("sigma_min", 0.001)
        self.sigma_max = sigma_bins_params.get("sigma_max", 1000.0)

        # Initialize EMA loss bins to zero, as the first run will populate them directly.
        self.ema_loss_bins = torch.zeros(self.sigma_num_bins, device=self.device)
        self.val_ema_loss_bins = torch.zeros(
            self.sigma_num_bins, device=self.device
        )
        # Counter for periodic p_lambda updates (only used after warm-up)
        self.sigma_step_count = torch.zeros(1, dtype=torch.long, device=self.device)
        self.iterations_since_start = 0
        self.first_validation_run = False
        self.ema_train_history = {}  # key: million step index, value: tensor of EMA training bins
        self.ema_val_history = {}
        
    @torch.no_grad()  
    def _bin_index_to_sigma(self,bin_idx):
        log_sigma_min = math.log(self.sigma_min)
        log_sigma_max = math.log(self.sigma_max)
        bin_width = (log_sigma_max - log_sigma_min) / self.sigma_num_bins
        
        bin_idx = torch.as_tensor(bin_idx, dtype=torch.float32)
        log_sigma = log_sigma_min + bin_idx * bin_width
        return torch.exp(log_sigma)

    @torch.no_grad()
    def _calculate_and_update_ema_bins(
        self, loss_term, ema_bins, bin_indices, is_validation=False
    ):
        """Helper to calculate and update EMA loss bins for a given loss tensor."""
        if loss_term is None or loss_term.numel() == 0:
            return ema_bins  # Return unmodified bins if loss term is empty

        # 1. Calculate Loss per Sample
        if loss_term.ndim > 1:
            dims_to_reduce = tuple(range(1, loss_term.ndim))
            per_sample_loss = loss_term.mean(dim=dims_to_reduce)
        else:
            per_sample_loss = loss_term
        # Shape: [B]

        # 2. Aggregate Loss per Bin for this Batch
        bin_sums = torch.zeros_like(ema_bins, device=self.device)
        bin_counts = torch.zeros_like(ema_bins, dtype=torch.long, device=self.device)
        bin_indices_long = bin_indices.long()
        bin_sums.scatter_add_(dim=0, index=bin_indices_long, src=per_sample_loss)
        one_vec = torch.ones_like(
            bin_indices_long, dtype=torch.long, device=self.device
        )
        bin_counts.scatter_add_(dim=0, index=bin_indices_long, src=one_vec)

        # 3. Synchronize Aggregates across GPUs (if using DDP)
        if self.is_distributed:
            dist.all_reduce(bin_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(bin_counts, op=dist.ReduceOp.SUM)

        # 4. Calculate Average Loss per Bin
        avg_loss_this_batch = bin_sums / (bin_counts + self.sigma_eps)

        # 5. Update EMA Loss Bins (Conditional Logic)
        update_mask = bin_counts > 0  # Bins sampled in this batch

        is_warmup = self.iterations_since_start < self.sigma_warmup_iterations
        if is_validation:
            if self.first_validation_run:
                # On the first validation, directly set the EMA to the calculated average.
                ema_bins[update_mask] = avg_loss_this_batch[update_mask]
            else:
                # For subsequent validations, apply the EMA formula.
                ema_bins[update_mask] = (
                    self.sigma_ema_decay * ema_bins[update_mask]
                    + (1.0 - self.sigma_ema_decay) * avg_loss_this_batch[update_mask]
                )
        elif is_warmup:
            # --- Phase 1: Warm-up ---
            ema_bins[update_mask] = avg_loss_this_batch[update_mask]
        else:
            # --- Phase 2: Adaptive EMA ---
            ema_bins[update_mask] = (
                self.sigma_ema_decay * ema_bins[update_mask]
                + (1.0 - self.sigma_ema_decay) * avg_loss_this_batch[update_mask]
            )

        return ema_bins
    
    @torch.no_grad()
    def _log_wandb_bar_plot(self, ema_bins, column_name, title, log_key, step_group=None):
        """Helper to log a single bar plot to wandb."""
        ema_loss_cpu = ema_bins.cpu().numpy()
        bin_indices_list = list(range(self.sigma_num_bins))
        ema_data = [[idx, self._bin_index_to_sigma(idx), ema_loss_cpu[idx]] for idx in bin_indices_list]

        ema_table = wandb.Table(data=ema_data, columns=["bin_index", "sigma", column_name])
        ema_bar_plot = wandb.plot.line(
            ema_table,
            "bin_index",
            column_name,
            title=title,
        )
        if step_group is not None:
            log_key = f"{log_key}_{step_group}_M"  # e.g., ema_loss_per_bin_bar_3_M
        wandb.log({log_key: ema_bar_plot}, step=self.iterations_since_start)


    @torch.no_grad()
    def _log_ema_plots(self,update_snapshot=False,million_step_idx=0):
        """
        Helper function to create and log combined training+validation EMA loss plot.
        Called from both training and validation EMA update functions.
        """
        if not (self.rank == 0 and self.log_to_wandb):
            return

        def create_plot(train_bins, val_bins, keys, title, log_key):
            if not (hasattr(self, train_bins) and hasattr(self, val_bins)):
                return

            train_ema_loss_cpu = getattr(self, train_bins).cpu().numpy()
            val_ema_loss_cpu = getattr(self, val_bins).cpu().numpy()

            bin_indices_list = list(range(self.sigma_num_bins))

            xs = bin_indices_list
            ys = [train_ema_loss_cpu.tolist(), val_ema_loss_cpu.tolist()]

            plot = wandb.plot.line_series(
                xs=xs, ys=ys, keys=keys, title=title, xname="Bin Index"
            )

            wandb.log({log_key: plot}, step=self.iterations_since_start)
        # def create_plot_snapshot(keys, title, log_key,m_step):
        #     if (m_step not in self.ema_train_history.keys()) or (m_step not in self.ema_val_history.keys()):
        #         return

        #     train_ema_loss_cpu = self.ema_train_history[m_step]
        #     val_ema_loss_cpu = self.ema_val_history[m_step]


        #     bin_indices_list = list(range(self.sigma_num_bins))

        #     xs = bin_indices_list
        #     ys = [train_ema_loss_cpu.tolist(), val_ema_loss_cpu.tolist()]

        #     plot = wandb.plot.line_series(
        #         xs=xs, ys=ys, keys=keys, title=title, xname="Bin Index"
        #     )

        #     wandb.log({log_key: plot}, step=self.iterations_since_start)



        def create_plot_snapshot(keys, title, log_key, m_step):
            if (m_step not in self.ema_train_history.keys()) or (m_step not in self.ema_val_history.keys()):
                return

            train_ema_loss_cpu = self.ema_train_history[m_step]
            val_ema_loss_cpu = self.ema_val_history[m_step]

            bin_indices_list = list(range(self.sigma_num_bins))

            xs = bin_indices_list
            ys_train = train_ema_loss_cpu.tolist()
            ys_val = val_ema_loss_cpu.tolist()

            # Larger figure
            fig, ax = plt.subplots(figsize=(8, 6))

            # Train = solid line
            ax.plot(xs, ys_train, label=keys[0], linestyle="-", linewidth=1.5)

            # Val = dashed line
            ax.plot(xs, ys_val, label=keys[1], linestyle="--",  linewidth=1.5)
            
            
            # # Train = solid line, smaller circle markers
            # ax.plot(xs, ys_train, label=keys[0], marker="o", linestyle="-", 
            #         markersize=4, linewidth=1.5)

            # # Val = dashed line, smaller circle markers
            # ax.plot(xs, ys_val, label=keys[1], marker="o", linestyle="--", 
            #         markersize=4, linewidth=1.5)

            # ax.set_title(title)
            ax.set_xlabel("Bin Index")
            ax.set_ylabel("EMA Loss")
            ax.legend()
            ax.grid(True)

            # Log to W&B as an image
            wandb.log({log_key: wandb.Image(fig,caption=title)}, step=self.iterations_since_start)

            plt.close(fig)

        try:
            # Combined Plot
            create_plot(
                "ema_loss_bins",
                "val_ema_loss_bins",
                ["(Combined) Training EMA Loss", "(Combined) Validation EMA Loss"],
                "(Combined) Training vs Validation EMA Loss per Log-Sigma Bin",
                "combined_train_val_ema_loss",
            )
            if update_snapshot:
                # for m_step, ema_bins_snapshot in sorted(self.ema_train_history.items()):
                    # Combined Plot
                print(f"plotting {million_step_idx} million steps")
                create_plot_snapshot(
                    ["(Combined) Training EMA Loss", "(Combined) Validation EMA Loss"],
                    f"(Combined) Training vs Validation EMA Loss per Log-Sigma Bin (after {million_step_idx} M steps)",
                    f"images/combined_train_val_ema_loss_M",
                    million_step_idx
                )

        except Exception as e:
            print(f"Warning: Failed to log combined EMA plot to wandb: {e}")


    @torch.no_grad()
    def _update_sigma_bin_state(
        self,
        loss_term_for_ema,
        bin_indices,
    ):
        """
        Updates EMA loss bins. During warmup, uses direct assignment.
        After warmup, uses EMA smoothing and updates p_lambda periodically.
        Handles DDP synchronization.
        """
        # --- Update Bins ---
        # Combined
        self.ema_loss_bins = self._calculate_and_update_ema_bins(
            loss_term_for_ema, self.ema_loss_bins, bin_indices
        )
        update_snapshot = False
        million_step_idx = self.iterations_since_start // 1000000
        if million_step_idx not in self.ema_train_history.keys():
            self.ema_train_history[million_step_idx] = self.ema_loss_bins.clone().detach().cpu().numpy().copy()
            update_snapshot = True


        # --- Periodic Logging ---
        if self.iterations_since_start >= self.sigma_warmup_iterations:
            self.sigma_step_count += 1
            current_count = self.sigma_step_count.item()

            if current_count >= self.sigma_update_every:
                self.sigma_step_count.zero_()

                if self.rank == 0 and self.log_to_wandb:
                    try:
                        # Log Combined
                        self._log_wandb_bar_plot(
                            self.ema_loss_bins,
                            "ema_loss",
                            "(Combined) EMA Loss per Log-Sigma Bin Index",
                            "ema_loss_per_bin_bar",
                        )
                        # # Log each historical snapshot
                        # for m_step, ema_bins_snapshot in sorted(self.ema_train_history.items()):
                        #     self._log_wandb_bar_plot(
                        #         ema_bins_snapshot, f"ema_loss_{m_step}_M",
                        #         f"(Combined) EMA Loss per Log-Sigma Bin (after {m_step}M steps)",
                        #         "ema_loss_per_bin_bar", step_group=m_step
                        #     )

                        # Update combined plot after training EMA update
                        self._log_ema_plots(update_snapshot,million_step_idx)

                    except Exception as e:
                        # Update warning message
                        print(f"Warning: Failed to log bar plots to wandb: {e}")
                # --- End Logging ---
                
    @torch.no_grad()
    def _update_and_log_val_sigma_bins(
        self,
        all_val_loss_terms,
        all_val_bin_indices,
    ):
        """
        Aggregates validation loss per sigma bin across batches, updates an EMA of these
        losses, and logs a plot to wandb. Handles DDP synchronization.
        """
        if not all_val_loss_terms or not hasattr(self, "val_ema_loss_bins"):
            return

        # 1. Concatenate all batch results from this rank
        loss_term_for_ema = torch.cat(all_val_loss_terms)
        bin_indices = torch.cat(all_val_bin_indices)
        """
        (Pdb) loss_term_for_ema.shape
        torch.Size([320, 4, 448, 448])
        (Pdb) bin_indices.shape
        torch.Size([320])
        """
        

        # --- Combined Loss ---
        self.val_ema_loss_bins = self._calculate_and_update_ema_bins(
            loss_term_for_ema, self.val_ema_loss_bins, bin_indices, is_validation=True
        )

        update_snapshot = False
        million_step_idx = self.iterations_since_start // 1000000
        if million_step_idx not in self.ema_val_history.keys():
            self.ema_val_history[million_step_idx] = self.val_ema_loss_bins.clone().detach().cpu().numpy().copy()
            update_snapshot = True

        if self.first_validation_run:
            self.first_validation_run = False

        # 7. Log plot to wandb from the main rank
        if self.rank == 0:
            try:
                # Log Combined
                self._log_wandb_bar_plot(
                    self.val_ema_loss_bins,
                    "val_ema_loss",
                    "(Combined) Validation EMA Loss per Log-Sigma Bin Index",
                    "val_ema_loss_per_bin_bar",
                )
                
                # # Log each historical snapshot
                # for m_step, ema_bins_snapshot in sorted(self.ema_val_history.items()):
                #     self._log_wandb_bar_plot(
                #         ema_bins_snapshot, f"ema_loss_{m_step}_M",
                #         f"(Combined) Validation EMA Loss per Log-Sigma Bin (after {m_step}M steps)",
                #         "val_ema_loss_per_bin_bar", step_group=m_step
                #     )

                # Update combined plot after validation EMA update
                self._log_ema_plots(update_snapshot,million_step_idx)
                

            except Exception as e:
                self.print0(
                    f"Warning: Failed to log validation sigma bins to wandb: {e}"
                )


