"""
LARESKT-V6.1: Per-step 可学习残差比例

V5.1 所有迭代步共享一个 step_scale 标量。
V6.1 为每步分配独立的可学习残差比例 step_scales[i]，让不同步学到不同的更新幅度：
- 浅层步（i 小）倾向于粗粒度更新（大 scale）
- 深层步（i 大）倾向于精细修正（小 scale）

初始化为全 1.0，退化为 V4.2-Fixed 行为。
"""
import torch
from torch import nn

from .lareskt_v5_1 import LARESKT_V5_1


class LARESKT_V6_1(LARESKT_V5_1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "lareskt_v6_1"
        # 替换单个 step_scale 为 per-step 向量
        del self.step_scale
        self.step_scales = nn.Parameter(torch.ones(self.mean_recurrence * 2 + 1))

    def recurrent_forward(self, pre_output, q_embed_data, qa_embed_prev=None,
                          num_steps=None, init_states=None, return_all_states=False):
        if init_states is None:
            states = self.initialize_state(pre_output)
        else:
            states = init_states

        if num_steps is None:
            num_steps = self.randomized_iteration_sampler()
        if isinstance(num_steps, torch.Tensor):
            num_steps = int(num_steps.item())
        num_steps = max(1, num_steps)

        all_step_states = []
        for step_idx in range(num_steps):
            prev_states = states

            states = self.fuse_state(states, pre_output)

            if qa_embed_prev is not None:
                gate = self.core_query_gate(torch.cat([states, qa_embed_prev], dim=-1))
                core_query = gate * states + (1 - gate) * qa_embed_prev
            else:
                core_query = states

            states = self.core_block(core_query, pre_output)

            # 残差更新：使用 per-step 可学习比例
            states = prev_states + self.step_scales[step_idx] * (states - prev_states)
            states = self.state_norm(states)

            all_step_states.append(states)

        d_output = states
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)

        if return_all_states:
            return preds, concat_q, all_step_states
        return preds, concat_q, None
