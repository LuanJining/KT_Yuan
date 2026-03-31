"""
LARESKT-V3: 改进pre_block的query设计

核心改动：pre_block的query从纯q_embed改为q_embed和qa_embed的门控融合，
让query也包含历史答题信息，使pre_block能更好地建模当前知识状态。

原始：pre_block(query=q_embed, kv=qa_embed)
V3：  pre_block(query=gate*q_embed + (1-gate)*qa_embed, kv=qa_embed)
"""
import torch
from torch import nn

from .lareskt import LARESKT


class LARESKT_V3(LARESKT):
    def __init__(self, *args, query_fusion="gate", **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "lareskt_v3"
        self.query_fusion = query_fusion

        if query_fusion == "gate":
            # 门控融合：学习q_embed和qa_embed的混合比例
            self.query_gate = nn.Sequential(
                nn.Linear(2 * self.d_model, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, 1),
                nn.Sigmoid()
            )
        elif query_fusion == "add":
            # 直接相加后投影
            self.query_proj = nn.Linear(self.d_model, self.d_model)
        elif query_fusion == "concat":
            # 拼接后投影
            self.query_proj = nn.Linear(2 * self.d_model, self.d_model)

    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()

        pid_data, q_data, target = self._build_input_triplet(q, c, r, qshft, cshft, rshft)
        q_embed_data, qa_embed_data = self._embed_inputs(pid_data, q_data, target)

        # 核心改动：融合q_embed和历史qa_embed作为pre_block的query
        # 将qa_embed右移一位，第一个位置用零填充，避免数据泄露
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
            query = q_embed_data  # 退化为原始

        pre_output = self.pre_block(query, qa_embed_data)
        preds, concat_q, _ = self.recurrent_forward(pre_output, q_embed_data)

        if train:
            sem_pre_output, sem_q_embed_data, sem_masks = None, None, None
            if all(k in dcur for k in ["sem_qseqs", "sem_cseqs", "sem_rseqs", "sem_shft_qseqs", "sem_shft_cseqs", "sem_shft_rseqs"]):
                sem_q = dcur["sem_qseqs"].long()
                sem_c = dcur["sem_cseqs"].long()
                sem_r = dcur["sem_rseqs"].long()
                sem_qshft = dcur["sem_shft_qseqs"].long()
                sem_cshft = dcur["sem_shft_cseqs"].long()
                sem_rshft = dcur["sem_shft_rseqs"].long()
                sem_masks = dcur.get("sem_masks")

                sem_pid_data, sem_q_data, sem_target = self._build_input_triplet(
                    sem_q, sem_c, sem_r, sem_qshft, sem_cshft, sem_rshft
                )
                sem_q_embed_data, sem_qa_embed_data = self._embed_inputs(sem_pid_data, sem_q_data, sem_target)

                sem_qa_embed_prev = torch.cat([
                    torch.zeros_like(sem_qa_embed_data[:, :1, :]),
                    sem_qa_embed_data[:, :-1, :]
                ], dim=1)

                if self.query_fusion == "gate":
                    sem_gate = self.query_gate(torch.cat([sem_q_embed_data, sem_qa_embed_prev], dim=-1))
                    sem_query = sem_gate * sem_q_embed_data + (1 - sem_gate) * sem_qa_embed_prev
                elif self.query_fusion == "add":
                    sem_query = self.query_proj(sem_q_embed_data + sem_qa_embed_prev)
                else:
                    sem_query = sem_q_embed_data

                sem_pre_output = self.pre_block(sem_query, sem_qa_embed_data)

            tla_loss, sla_loss = self.calc_alignment_losses(
                pre_output, q_embed_data,
                masks=dcur.get("masks"),
                sem_pre_output=sem_pre_output,
                sem_q_embed_data=sem_q_embed_data if sem_pre_output is not None else None,
                sem_masks=sem_masks,
            )
            return preds, tla_loss, sla_loss

        if qtest:
            return preds, concat_q
        return preds
