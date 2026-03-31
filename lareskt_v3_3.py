"""
LARESKT-V3.3: core_block的query也加入历史感知

在V3基础上，core_block每步迭代的query（states）也融入历史qa信息，
让每步迭代能感知到历史答题情况，而不只是当前的states。
"""
import torch
from torch import nn

from .lareskt_v3 import LARESKT_V3


class LARESKT_V3_3(LARESKT_V3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "lareskt_v3_3"

        # core_block的query融合门控
        self.core_query_gate = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )

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
            states = self.fuse_state(states, pre_output)

            # 核心改动：core_block的query融合历史qa信息
            if qa_embed_prev is not None:
                gate = self.core_query_gate(torch.cat([states, qa_embed_prev], dim=-1))
                core_query = gate * states + (1 - gate) * qa_embed_prev
            else:
                core_query = states

            states = self.core_block(core_query, pre_output)
            all_step_states.append(states)

        d_output = states
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)

        if return_all_states:
            return preds, concat_q, all_step_states
        return preds, concat_q, None

    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()

        pid_data, q_data, target = self._build_input_triplet(q, c, r, qshft, cshft, rshft)
        q_embed_data, qa_embed_data = self._embed_inputs(pid_data, q_data, target)

        # pre_block的query改进（继承自V3）
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
        # core_block也传入qa_embed_prev
        preds, concat_q, _ = self.recurrent_forward(pre_output, q_embed_data, qa_embed_prev=qa_embed_prev)

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
                sem_gate = self.query_gate(torch.cat([sem_q_embed_data, sem_qa_embed_prev], dim=-1))
                sem_query = sem_gate * sem_q_embed_data + (1 - sem_gate) * sem_qa_embed_prev
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
