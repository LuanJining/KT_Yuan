import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .lareskt import LARESKT


class StepAllocator(nn.Module):
    """轻量级的步数分配器"""
    def __init__(self, d_model, hidden_dim=128, min_steps=1, max_steps=5):
        super().__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.num_steps = max_steps - min_steps + 1

        # 简单的MLP预测步数分布
        self.predictor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, self.num_steps)
        )

        # 用于计算序列级别的特征（取平均或最后一个）
        self.pooling_type = "mean"  # 可选: "mean", "last", "max"

    def forward(self, hidden_states, mask=None):
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model) - pre_block的输出
            mask: (batch_size, seq_len) - 有效位置的mask
        Returns:
            step_logits: (batch_size, num_steps) - 每个步数的logits
            step_probs: (batch_size, num_steps) - 每个步数的概率
        """
        # 序列级别的特征提取
        if self.pooling_type == "mean":
            if mask is not None:
                # 确保mask和hidden_states维度匹配
                seq_len = hidden_states.size(1)
                if mask.size(1) != seq_len:
                    # 如果mask比hidden_states短，在前面补1
                    if mask.size(1) < seq_len:
                        pad_size = seq_len - mask.size(1)
                        mask = torch.cat([torch.ones(mask.size(0), pad_size, device=mask.device, dtype=mask.dtype), mask], dim=1)
                    else:
                        # 如果mask比hidden_states长，截断
                        mask = mask[:, :seq_len]

                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = hidden_states.mean(dim=1)
        elif self.pooling_type == "last":
            if mask is not None:
                # 调整mask维度
                seq_len = hidden_states.size(1)
                if mask.size(1) != seq_len:
                    if mask.size(1) < seq_len:
                        pad_size = seq_len - mask.size(1)
                        mask = torch.cat([torch.ones(mask.size(0), pad_size, device=mask.device, dtype=mask.dtype), mask], dim=1)
                    else:
                        mask = mask[:, :seq_len]

                lengths = mask.sum(dim=1).long() - 1
                lengths = lengths.clamp(min=0, max=seq_len-1)
                pooled = hidden_states[torch.arange(hidden_states.size(0)), lengths]
            else:
                pooled = hidden_states[:, -1]
        elif self.pooling_type == "max":
            pooled = hidden_states.max(dim=1)[0]

        # 预测步数分布
        step_logits = self.predictor(pooled)
        step_probs = F.softmax(step_logits, dim=-1)

        return step_logits, step_probs

    def sample_steps(self, step_probs, deterministic=False):
        """
        从概率分布中采样步数
        Args:
            step_probs: (batch_size, num_steps)
            deterministic: 是否使用确定性策略（选择概率最大的）
        Returns:
            steps: (batch_size,) - 采样的步数
        """
        if deterministic:
            steps = step_probs.argmax(dim=-1)
        else:
            steps = torch.multinomial(step_probs, num_samples=1).squeeze(-1)

        # 转换为实际步数 (1-5)
        actual_steps = steps + self.min_steps
        return actual_steps, steps  # 返回实际步数和索引


class LARESKT_RL(LARESKT):
    """带强化学习步数分配的LARES-KT"""
    def __init__(
        self,
        n_question,
        n_pid,
        d_model,
        n_blocks,
        dropout,
        d_ff=256,
        num_attn_heads=8,
        seq_len=200,
        kq_same=1,
        final_fc_dim=512,
        final_fc_dim2=256,
        separate_qa=False,
        emb_type="qid",
        emb_path="",
        mean_recurrence=3,
        sampling_scheme="uniform",
        state_init_method="zero",
        state_std=0.02,
        state_scale=1.0,
        adapter_type="concat",
        tau=0.2,
        alpha=0.1,
        gamma=0.1,
        same_step=True,
        # RL相关参数
        use_rl_allocator=True,
        allocator_hidden_dim=128,
        min_steps=1,
        max_steps=5,
        efficiency_weight=0.01,  # 效率损失权重
        entropy_weight=0.01,     # 熵正则化权重
        **kwargs,
    ):
        super().__init__(
            n_question=n_question,
            n_pid=n_pid,
            d_model=d_model,
            n_blocks=n_blocks,
            dropout=dropout,
            d_ff=d_ff,
            num_attn_heads=num_attn_heads,
            seq_len=seq_len,
            kq_same=kq_same,
            final_fc_dim=final_fc_dim,
            final_fc_dim2=final_fc_dim2,
            separate_qa=separate_qa,
            emb_type=emb_type,
            emb_path=emb_path,
            mean_recurrence=mean_recurrence,
            sampling_scheme=sampling_scheme,
            state_init_method=state_init_method,
            state_std=state_std,
            state_scale=state_scale,
            adapter_type=adapter_type,
            tau=tau,
            alpha=alpha,
            gamma=gamma,
            same_step=same_step,
            **kwargs,
        )

        self.model_name = "lareskt_rl"
        self.use_rl_allocator = use_rl_allocator
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.efficiency_weight = efficiency_weight
        self.entropy_weight = entropy_weight

        if self.use_rl_allocator:
            self.step_allocator = StepAllocator(
                d_model=d_model,
                hidden_dim=allocator_hidden_dim,
                min_steps=min_steps,
                max_steps=max_steps
            )

    def forward(self, dcur, qtest=False, train=False):
        """
        前向传播，支持RL步数分配
        """
        # 构建输入
        q = dcur["qseqs"].long()
        c = dcur["cseqs"].long()
        r = dcur["rseqs"].long()
        qshft = dcur["shft_qseqs"].long()
        cshft = dcur["shft_cseqs"].long()
        rshft = dcur["shft_rseqs"].long()

        pid_data, q_data, target = self._build_input_triplet(
            q, c, r, qshft, cshft, rshft
        )

        # Embedding
        q_embed_data, qa_embed_data = self._embed_inputs(pid_data, q_data, target)

        # Pre-block处理
        pre_output = self.pre_block(q_embed_data, qa_embed_data)

        # 步数分配
        if self.use_rl_allocator and train:
            # 训练模式：使用RL分配器
            masks = dcur.get("masks")
            step_logits, step_probs = self.step_allocator(pre_output, masks)

            # 采样步数
            allocated_steps, step_indices = self.step_allocator.sample_steps(
                step_probs, deterministic=False
            )

            # 对batch中的每个样本使用不同的步数
            batch_size = pre_output.size(0)
            all_preds = []
            all_concat_q = []

            for i in range(batch_size):
                num_steps = int(allocated_steps[i].item())
                sample_pre_output = pre_output[i:i+1]
                sample_q_embed = q_embed_data[i:i+1]

                preds, concat_q, _ = self.recurrent_forward(
                    sample_pre_output,
                    sample_q_embed,
                    num_steps=num_steps,
                    return_all_states=False
                )
                all_preds.append(preds)
                all_concat_q.append(concat_q)

            preds = torch.cat(all_preds, dim=0)
            concat_q = torch.cat(all_concat_q, dim=0)

            # 计算RL相关的额外信息
            rl_info = {
                "step_logits": step_logits,
                "step_probs": step_probs,
                "allocated_steps": allocated_steps,
                "step_indices": step_indices
            }

            return preds, rl_info

        else:
            # 测试模式：每个样本用自己的最优步数（确定性策略）
            if self.use_rl_allocator:
                masks = dcur.get("masks")
                _, step_probs = self.step_allocator(pre_output, masks)
                allocated_steps, _ = self.step_allocator.sample_steps(
                    step_probs, deterministic=True
                )

                batch_size = pre_output.size(0)
                all_preds = []
                all_concat_q = []

                for i in range(batch_size):
                    num_steps = int(allocated_steps[i].item())
                    preds_i, concat_q_i, _ = self.recurrent_forward(
                        pre_output[i:i+1],
                        q_embed_data[i:i+1],
                        num_steps=num_steps,
                        return_all_states=False
                    )
                    all_preds.append(preds_i)
                    all_concat_q.append(concat_q_i)

                preds = torch.cat(all_preds, dim=0)
                concat_q = torch.cat(all_concat_q, dim=0)
            else:
                preds, concat_q, _ = self.recurrent_forward(
                    pre_output,
                    q_embed_data,
                    num_steps=self.mean_recurrence,
                    return_all_states=False
                )

            if qtest:
                return preds, concat_q
            return preds

    def compute_rl_loss(self, rl_info, task_loss, masks=None):
        """
        计算强化学习损失

        Args:
            rl_info: forward返回的RL信息
            task_loss: 主任务损失（BCE loss），shape: (batch_size, seq_len)
            masks: 有效位置mask

        Returns:
            rl_loss: 强化学习损失
            loss_dict: 详细的损失信息
        """
        step_logits = rl_info["step_logits"]
        step_probs = rl_info["step_probs"]
        allocated_steps = rl_info["allocated_steps"]
        step_indices = rl_info["step_indices"]

        # 1. 计算reward
        # Reward = -task_loss - efficiency_penalty
        # task_loss越小越好，步数越少越好
        if masks is not None:
            # 只计算有效位置的平均loss
            valid_loss = (task_loss * masks).sum(dim=1) / masks.sum(dim=1).clamp(min=1)
        else:
            valid_loss = task_loss.mean(dim=1)

        # 效率惩罚：步数越多，惩罚越大
        efficiency_penalty = (allocated_steps.float() - self.min_steps) / (self.max_steps - self.min_steps)

        # 总reward（负的，因为我们要最小化loss）
        rewards = -valid_loss - self.efficiency_weight * efficiency_penalty

        # 2. Policy Gradient Loss (REINFORCE)
        # 使用baseline减少方差（这里用batch均值）
        baseline = rewards.mean().detach()
        advantages = rewards - baseline

        # 计算log概率
        log_probs = F.log_softmax(step_logits, dim=-1)
        selected_log_probs = log_probs.gather(1, step_indices.unsqueeze(-1)).squeeze(-1)

        # Policy gradient loss
        pg_loss = -(selected_log_probs * advantages.detach()).mean()

        # 3. 熵正则化（鼓励探索）
        entropy = -(step_probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.entropy_weight * entropy

        # 总RL损失
        rl_loss = pg_loss + entropy_loss

        loss_dict = {
            "rl_loss": rl_loss.item(),
            "pg_loss": pg_loss.item(),
            "entropy": entropy.item(),
            "avg_reward": rewards.mean().item(),
            "avg_steps": allocated_steps.float().mean().item(),
            "efficiency_penalty": efficiency_penalty.mean().item()
        }

        return rl_loss, loss_dict
