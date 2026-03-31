"""
LARESKT-V7.2: 去除对齐损失，纯BCE训练

在 V7.1 基础上，完全移除 alignment loss 的计算。
当 alpha=0 且 gamma=0 时，V7.1 的 calc_alignment_losses 仍然会
执行 2 次 recurrent_forward（每次 ~3.5 步 core_block），
白白浪费约 60% 的计算量。V7.2 直接跳过这部分，大幅加速训练。

仅保留主路径：embedding -> pre_block -> recurrent_forward -> output
"""
import torch
from torch import nn

from .lareskt_v7_1 import LARESKT_V7_1


class LARESKT_V7_2(LARESKT_V7_1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "lareskt_v7_2"

    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()

        pid_data, q_data, target = self._build_input_triplet(q, c, r, qshft, cshft, rshft)
        q_embed_data, qa_embed_data = self._embed_inputs(pid_data, q_data, target)

        qa_embed_prev = torch.cat([
            torch.zeros_like(qa_embed_data[:, :1, :]),
            qa_embed_data[:, :-1, :]
        ], dim=1)

        if self.query_fusion == "gate":
            gate = self.query_gate(torch.cat([q_embed_data, qa_embed_prev], dim=-1))
            query = gate * q_embed_data + (1 - gate) * qa_embed_prev
        elif self.query_fusion == "add":
            query = self.query_proj(q_embed_data + qa_embed_prev)
        elif self.query_fusion == "concat":
            query = self.query_proj(torch.cat([q_embed_data, qa_embed_prev], dim=-1))
        else:
            query = q_embed_data

        pre_output = self.pre_block(query, qa_embed_data)
        preds, concat_q, _ = self.recurrent_forward(pre_output, q_embed_data, qa_embed_prev=qa_embed_prev)

        if train:
            return preds, 0.0, 0.0

        if qtest:
            return preds, concat_q
        return preds
