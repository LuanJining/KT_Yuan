"""
LARESKT-V3.1: 用前k步加权平均作为历史摘要

在V3基础上，query不只融合前一步qa_embed，
而是用前k步的指数加权平均（越近的步权重越大），
让query感知更长的历史趋势。
"""
import torch
from torch import nn

from .lareskt import LARESKT


class LARESKT_V3_1(LARESKT):
    def __init__(self, *args, history_window=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "lareskt_v3_1"
        self.history_window = history_window

        # 对历史k步的qa_embed做加权融合
        self.history_weights = nn.Parameter(torch.ones(history_window))  # 可学习权重

        # 门控：融合q_embed和历史摘要
        self.query_gate = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )

    def _build_history_summary(self, qa_embed_data):
        """
        对每个位置t，计算前k步qa_embed的加权平均
        qa_embed_data: (B, T, D)
        返回: (B, T, D)
        """
        B, T, D = qa_embed_data.shape
        k = self.history_window
        weights = torch.softmax(self.history_weights, dim=0)  # (k,) 归一化

        # 对每个位置t，收集前k步（不含当前步，避免泄露）
        # 用padding方式实现：在序列前面补k个零
        padded = torch.cat([
            torch.zeros(B, k, D, device=qa_embed_data.device, dtype=qa_embed_data.dtype),
            qa_embed_data
        ], dim=1)  # (B, T+k, D)

        # 对每个位置t，取padded[t:t+k]（即原始序列的t-k到t-1步）
        # 用unfold实现滑动窗口
        # padded[:, :-1, :] 去掉最后一个（当前步），取前T+k-1步
        windows = padded[:, :-1, :].unfold(1, k, 1)  # (B, T, D, k)
        windows = windows.permute(0, 1, 3, 2)  # (B, T, k, D)

        # 加权求和
        history_summary = (windows * weights.view(1, 1, k, 1)).sum(dim=2)  # (B, T, D)
        return history_summary

    def _build_query(self, q_embed_data, qa_embed_data):
        history = self._build_history_summary(qa_embed_data)
        gate = self.query_gate(torch.cat([q_embed_data, history], dim=-1))
        return gate * q_embed_data + (1 - gate) * history

    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()

        pid_data, q_data, target = self._build_input_triplet(q, c, r, qshft, cshft, rshft)
        q_embed_data, qa_embed_data = self._embed_inputs(pid_data, q_data, target)

        query = self._build_query(q_embed_data, qa_embed_data)
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
                sem_query = self._build_query(sem_q_embed_data, sem_qa_embed_data)
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
