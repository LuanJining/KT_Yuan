"""
LARESKT-V5.1: 迭代间残差改进

在每步迭代周围加入残差连接，让每步只做增量更新而非完全替换：

    prev_states = states
    states = fuse_state(states, pre_output)     # 不含 state_norm
    states = core_block(core_query, pre_output)
    states = prev_states + step_scale * (states - prev_states)  # 残差更新
    states = state_norm(states)

step_scale 是可学习标量，初始化为1.0（退化为原始行为，训练中自动学习最优更新幅度）。
"""
import torch
from torch import nn

from .lareskt_v4_2_fixed import LARESKT_V4_2_Fixed


class LARESKT_V5_1(LARESKT_V4_2_Fixed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "lareskt_v5_1"
        # 可学习标量，初始化为1.0（退化为原始行为）
        self.step_scale = nn.Parameter(torch.tensor(1.0))

    def fuse_state(self, states, pre_output):
        """重写 fuse_state：不做 state_norm，norm 移到残差更新之后"""
        if self.adapter_type == "concat":
            states = self.adapter(torch.cat([states, pre_output], dim=-1))
        elif self.adapter_type == "add":
            states = (states + pre_output) / 2
        elif self.adapter_type == "linear":
            gate = torch.sigmoid(self.adapter)
            states = gate * states + (1 - gate) * pre_output
        return states

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
        for _ in range(num_steps):
            prev_states = states

            states = self.fuse_state(states, pre_output)

            if qa_embed_prev is not None:
                gate = self.core_query_gate(torch.cat([states, qa_embed_prev], dim=-1))
                core_query = gate * states + (1 - gate) * qa_embed_prev
            else:
                core_query = states

            states = self.core_block(core_query, pre_output)

            # 残差更新：只做增量修改
            states = prev_states + self.step_scale * (states - prev_states)
            states = self.state_norm(states)

            all_step_states.append(states)

        d_output = states
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)

        if return_all_states:
            return preds, concat_q, all_step_states
        return preds, concat_q, None
