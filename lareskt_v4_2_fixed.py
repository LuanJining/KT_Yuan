"""
LARESKT-V4.2-Fixed: 修复V3.3中对齐损失路径与主路径不一致的bug

Bug描述：
  V3.3的calc_alignment_losses继承自基类LARESKT，调用recurrent_forward时
  没有传入qa_embed_prev，导致对齐损失路径退化为无历史感知版本，
  而主路径有历史感知。训练时两条路径不一致，对齐损失梯度方向错误。

修复：
  重写calc_alignment_losses，透传qa_embed_prev到recurrent_forward。
  同时forward中sem分支也正确传入sem_qa_embed_prev。
"""
import torch
from torch import nn

from .lareskt_v3_3 import LARESKT_V3_3


class LARESKT_V4_2_Fixed(LARESKT_V3_3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "lareskt_v4_2_fixed"

    def calc_alignment_losses(self, pre_output, q_embed_data, qa_embed_prev=None,
                               masks=None, sem_pre_output=None, sem_q_embed_data=None,
                               sem_qa_embed_prev=None, sem_masks=None):
        """修复版：对齐损失路径与主路径保持一致，均传入qa_embed_prev"""
        n_step = self.randomized_iteration_sampler() if self.same_step else None

        _, _, all_states_1 = self.recurrent_forward(
            pre_output, q_embed_data,
            qa_embed_prev=qa_embed_prev,
            num_steps=n_step,
            return_all_states=True,
        )

        if sem_pre_output is not None and sem_q_embed_data is not None:
            _, _, all_states_2 = self.recurrent_forward(
                sem_pre_output, sem_q_embed_data,
                qa_embed_prev=sem_qa_embed_prev,
                num_steps=n_step if self.same_step else None,
                return_all_states=True,
            )
        else:
            _, _, all_states_2 = self.recurrent_forward(
                pre_output, q_embed_data,
                qa_embed_prev=qa_embed_prev,
                num_steps=n_step if self.same_step else None,
                return_all_states=True,
            )

        final_1 = all_states_1[-1]
        final_2 = all_states_2[-1]

        seq_mask_1 = self._sequence_mask(masks)
        seq_mask_2 = self._sequence_mask(sem_masks if sem_masks is not None else masks)
        pair_mask = None
        if seq_mask_1 is not None and seq_mask_2 is not None:
            pair_mask = seq_mask_1 & seq_mask_2

        tla_loss = self._masked_infonce_loss(final_1, final_2, pair_mask)

        sla_loss = torch.zeros((), device=final_1.device)
        if len(all_states_1) > 1:
            idx = torch.randint(0, len(all_states_1) - 1, (1,), device=final_1.device).item()
            mid_state = all_states_1[idx]
            sla_loss = self._masked_infonce_loss(final_1, mid_state, seq_mask_1)

        return tla_loss, sla_loss

    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()

        pid_data, q_data, target = self._build_input_triplet(q, c, r, qshft, cshft, rshft)
        q_embed_data, qa_embed_data = self._embed_inputs(pid_data, q_data, target)

        qa_embed_prev = torch.cat([
            torch.zeros_like(qa_embed_data[:, :1, :]),
            qa_embed_data[:, :-1, :]
        ], dim=1)

        gate = self.query_gate(torch.cat([q_embed_data, qa_embed_prev], dim=-1))
        query = gate * q_embed_data + (1 - gate) * qa_embed_prev

        pre_output = self.pre_block(query, qa_embed_data)
        preds, concat_q, _ = self.recurrent_forward(pre_output, q_embed_data, qa_embed_prev=qa_embed_prev)

        if train:
            sem_pre_output, sem_q_embed_data, sem_qa_embed_prev, sem_masks = None, None, None, None
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
                sem_gate = self.query_gate(torch.cat([sem_q_embed_data, sem_qa_embed_prev], dim=-1))
                sem_query = sem_gate * sem_q_embed_data + (1 - sem_gate) * sem_qa_embed_prev
                sem_pre_output = self.pre_block(sem_query, sem_qa_embed_data)

            tla_loss, sla_loss = self.calc_alignment_losses(
                pre_output, q_embed_data,
                qa_embed_prev=qa_embed_prev,
                masks=dcur.get("masks"),
                sem_pre_output=sem_pre_output,
                sem_q_embed_data=sem_q_embed_data if sem_pre_output is not None else None,
                sem_qa_embed_prev=sem_qa_embed_prev if sem_pre_output is not None else None,
                sem_masks=sem_masks,
            )
            return preds, tla_loss, sla_loss

        if qtest:
            return preds, concat_q
        return preds
