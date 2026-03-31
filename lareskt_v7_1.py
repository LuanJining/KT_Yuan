"""
LARESKT-V7.1: Step Position Encoding（层位置编码）

问题：V6.2 的每步迭代输入相同的 pre_output，模型缺少"我在第几步"的感知，
导致所有步行为趋同，第一步就收敛，多步思考名存实亡。

改进：引入可学习的 step embedding，让模型知道当前处于第几步推理，
从而学到不同步的差异化行为（如：第1步粗粒度分析，第2步精细修正）。
类比 LLM 中不同层学到的不同表征——虽然各层参数相同，
但输入中隐含的层位置信息使各层行为不同。V7.1 将这一信息显式化。

    step_emb = step_embeddings(step_idx)       # 可学习的步位置向量
    states = fuse_state(states, pre_output + step_emb)  # 注入步位置信息
    ...
    gate = sigmoid(update_gate(states))         # 保留 V6.2 的动态残差门控
    states = prev_states + gate * (states - prev_states)
"""
import torch
from torch import nn

from .lareskt_v6_2 import LARESKT_V6_2


class LARESKT_V7_1(LARESKT_V6_2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "lareskt_v7_1"
        self.max_steps = self.mean_recurrence * 2 + 1
        self.step_embeddings = nn.Embedding(self.max_steps, self.d_model)

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

            # 注入步位置信息：不同步看到不同的 pre_output
            step_emb = self.step_embeddings(
                torch.tensor(step_idx, device=pre_output.device)
            )
            states = self.fuse_state(states, pre_output + step_emb)

            if qa_embed_prev is not None:
                gate = self.core_query_gate(torch.cat([states, qa_embed_prev], dim=-1))
                core_query = gate * states + (1 - gate) * qa_embed_prev
            else:
                core_query = states

            states = self.core_block(core_query, pre_output)

            # 动态残差更新
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
