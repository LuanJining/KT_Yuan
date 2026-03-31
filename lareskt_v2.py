"""
LARESKT-V2: 改进core_block的信息利用方式

核心改动：recurrent_forward中每步迭代时，
core_block的key/value不再是固定的pre_output，
而是pre_output和上一步states的门控融合，
让模型能感知跨步的状态变化。
"""
import torch
from torch import nn

from .lareskt import LARESKT


class LARESKT_V2(LARESKT):
    def __init__(self, *args, cross_step_gate=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "lareskt_v2"
        self.cross_step_gate = cross_step_gate

        if cross_step_gate:
            # 可学习的门控：控制pre_output和prev_states的融合比例
            # 输入：[pre_output; prev_states] -> gate scalar per position
            self.step_gate = nn.Sequential(
                nn.Linear(2 * self.d_model, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, 1),
                nn.Sigmoid()
            )

    def fuse_kv(self, pre_output, prev_states):
        """
        融合pre_output和prev_states作为core_block的key/value
        gate控制两者比例：gate=1时完全用pre_output（退化为原始lareskt）
        """
        if not self.cross_step_gate:
            return pre_output

        gate_input = torch.cat([pre_output, prev_states], dim=-1)
        gate = self.step_gate(gate_input)  # (B, T, 1)
        return gate * pre_output + (1 - gate) * prev_states

    def recurrent_forward(self, pre_output, q_embed_data, num_steps=None, init_states=None, return_all_states=False):
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
        prev_states = states  # 第一步prev_states=初始零状态

        for _ in range(num_steps):
            states = self.fuse_state(states, pre_output)
            # 关键改动：key/value是pre_output和prev_states的门控融合
            kv = self.fuse_kv(pre_output, prev_states)
            states = self.core_block(states, kv)
            prev_states = states
            all_step_states.append(states)

        d_output = states
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)

        if return_all_states:
            return preds, concat_q, all_step_states
        return preds, concat_q, None
