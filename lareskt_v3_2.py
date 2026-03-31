"""
LARESKT-V3.2: 答对/答错条件门控

在V3基础上，显式区分答对和答错对query的不同影响：
- 答对时：query偏向题目信息（知识状态稳定）
- 答错时：query更多融入历史交互信息（知识状态需要更新）
"""
import torch
from torch import nn

from .lareskt import LARESKT


class LARESKT_V3_2(LARESKT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "lareskt_v3_2"

        # 答对时的门控
        self.gate_correct = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        # 答错时的门控
        self.gate_wrong = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )

    def _build_query(self, q_embed_data, qa_embed_data, target):
        """
        target: (B, T) 答题结果，0或1
        """
        # 历史答题结果（右移一步，避免泄露）
        prev_target = torch.cat([
            torch.zeros_like(target[:, :1]),
            target[:, :-1]
        ], dim=1).float()  # (B, T)

        # 历史qa_embed（右移一步）
        qa_embed_prev = torch.cat([
            torch.zeros_like(qa_embed_data[:, :1, :]),
            qa_embed_data[:, :-1, :]
        ], dim=1)  # (B, T, D)

        feat = torch.cat([q_embed_data, qa_embed_prev], dim=-1)  # (B, T, 2D)

        # 分别计算答对/答错的gate
        gate_c = self.gate_correct(feat)  # (B, T, 1)
        gate_w = self.gate_wrong(feat)    # (B, T, 1)

        # 根据上一步答题结果选择gate
        prev_correct = prev_target.unsqueeze(-1)  # (B, T, 1)
        gate = prev_correct * gate_c + (1 - prev_correct) * gate_w

        return gate * q_embed_data + (1 - gate) * qa_embed_prev

    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()

        pid_data, q_data, target = self._build_input_triplet(q, c, r, qshft, cshft, rshft)
        q_embed_data, qa_embed_data = self._embed_inputs(pid_data, q_data, target)

        query = self._build_query(q_embed_data, qa_embed_data, target)
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
                sem_query = self._build_query(sem_q_embed_data, sem_qa_embed_data, sem_target)
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
