"""
LARESKT-V6.2: 动态残差门控（模型自主思考）

V5.1 的 step_scale 是全局固定标量，与输入无关，无法体现"自主思考"。
V6.2 将固定标量替换为输入相关的门控网络，让模型根据当前学生状态
动态决定每步接受多少新计算：

    gate = sigmoid(update_gate(states))
    states = prev_states + gate * (states - prev_states)

模型看到当前状态后自己判断"这次推理结果采信多少"，
不同的学生、不同的时刻、不同的迭代步，gate 都不同。
"""
import torch
from torch import nn

from .lareskt_v5_1 import LARESKT_V5_1


class LARESKT_V6_2(LARESKT_V5_1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "lareskt_v6_2"
        # 删掉 V5.1 的固定标量
        del self.step_scale
        # 动态门控：根据当前状态决定更新比例
        self.update_gate = nn.Linear(self.d_model, 1)

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

            # 动态残差更新：模型根据状态自主决定采信比例
            gate = torch.sigmoid(self.update_gate(states))
            states = prev_states + gate * (states - prev_states)
            states = self.state_norm(states)

            all_step_states.append(states)

        d_output = states
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)

        if return_all_states:
            return preds, concat_q, all_step_states
        return preds, concat_q, None
