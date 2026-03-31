import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .simplekt import Architecture


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LARESKT(nn.Module):
    def __init__(
        self,
        n_question,
        n_pid,
        d_model,
        n_blocks,
        dropout,
        d_ff=256,
        num_attn_heads=8,
        seq_len=200,
        kq_same=1,
        final_fc_dim=512,
        final_fc_dim2=256,
        separate_qa=False,
        emb_type="qid",
        emb_path="",
        mean_recurrence=3,
        sampling_scheme="uniform",
        state_init_method="zero",
        state_std=0.02,
        state_scale=1.0,
        adapter_type="concat",
        tau=0.2,
        alpha=0.1,
        gamma=0.1,
        same_step=True,
        **kwargs,
    ):
        super().__init__()
        self.model_name = "lareskt"
        print(f"model_name: {self.model_name}, emb_type: {emb_type}")

        self.n_question = n_question
        self.n_pid = n_pid
        self.dropout = dropout
        self.kq_same = kq_same
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.d_model = d_model

        self.mean_recurrence = mean_recurrence
        self.sampling_scheme = sampling_scheme
        self.state_init_method = state_init_method
        self.state_std = state_std
        self.state_scale = state_scale
        self.adapter_type = adapter_type
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.same_step = same_step

        embed_l = d_model

        if self.n_pid > 0:
            if emb_type.find("scalar") != -1:
                self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            else:
                self.difficult_param = nn.Embedding(self.n_pid + 1, embed_l)
            self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)

        if emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa:
                self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)

        self.pre_block = Architecture(
            n_question=n_question,
            n_blocks=n_blocks,
            n_heads=num_attn_heads,
            dropout=dropout,
            d_model=d_model,
            d_feature=d_model / num_attn_heads,
            d_ff=d_ff,
            kq_same=self.kq_same,
            model_type="simplekt",
            seq_len=seq_len,
            apply_pos_emb=True,
        )

        self.core_block = Architecture(
            n_question=n_question,
            n_blocks=n_blocks,
            n_heads=num_attn_heads,
            dropout=dropout,
            d_model=d_model,
            d_feature=d_model / num_attn_heads,
            d_ff=d_ff,
            kq_same=self.kq_same,
            model_type="simplekt",
            seq_len=seq_len,
            apply_pos_emb=False,
        )

        if self.adapter_type == "concat":
            self.adapter = nn.Linear(2 * d_model, d_model)
        elif self.adapter_type == "add":
            self.adapter = None
        elif self.adapter_type == "linear":
            self.adapter = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        else:
            raise ValueError(f"Unknown adapter type: {self.adapter_type}")

        self.state_norm = nn.LayerNorm(d_model)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1),
        )

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.dim() > 0 and self.n_pid > 0 and p.size(0) == self.n_pid + 1:
                torch.nn.init.constant_(p, 0.0)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    @torch.no_grad()
    def randomized_iteration_sampler(self):
        if self.training:
            if "uniform" in self.sampling_scheme:
                t = torch.randint(low=1, high=max(2, 1 + self.mean_recurrence * 2), size=(1,))
            elif "poisson-lognormal" in self.sampling_scheme:
                sigma = 0.5
                mu = math.log(max(1, self.mean_recurrence)) - (sigma ** 2 / 2)
                rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma)
                t = torch.poisson(torch.tensor([rate], dtype=torch.float)) + 1
                t = torch.minimum(t, torch.as_tensor(max(1, 3 * self.mean_recurrence)))
            elif "poisson-unbounded" in self.sampling_scheme:
                t = torch.poisson(torch.tensor([self.mean_recurrence], dtype=torch.float))
            elif "poisson-bounded" in self.sampling_scheme:
                t = torch.minimum(
                    torch.poisson(torch.tensor([self.mean_recurrence], dtype=torch.float)),
                    torch.as_tensor(max(1, 2 * self.mean_recurrence)),
                )
            elif "non-recurrent" in self.sampling_scheme:
                t = torch.as_tensor(1)
            else:
                t = torch.as_tensor(max(1, self.mean_recurrence))
        else:
            t = torch.as_tensor(max(1, self.mean_recurrence))

        return t.squeeze().to(dtype=torch.long)

    def initialize_state(self, hidden_states):
        x = torch.zeros_like(hidden_states)
        if self.state_init_method == "normal":
            torch.nn.init.trunc_normal_(x, mean=0.0, std=self.state_std, a=-3 * self.state_std, b=3 * self.state_std)
        return self.state_scale * x

    def fuse_state(self, states, pre_output):
        if self.adapter_type == "concat":
            states = self.adapter(torch.cat([states, pre_output], dim=-1))
        elif self.adapter_type == "add":
            states = (states + pre_output) / 2
        elif self.adapter_type == "linear":
            gate = torch.sigmoid(self.adapter)
            states = gate * states + (1 - gate) * pre_output
        return self.state_norm(states)

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
        for _ in range(num_steps):
            states = self.fuse_state(states, pre_output)
            states = self.core_block(states, pre_output)
            all_step_states.append(states)

        d_output = states
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)

        if return_all_states:
            return preds, concat_q, all_step_states
        return preds, concat_q, None

    def _sequence_mask(self, masks):
        if masks is None:
            return None
        if masks.dim() == 1:
            masks = masks.unsqueeze(0)
        head = torch.ones((masks.size(0), 1), dtype=masks.dtype, device=masks.device)
        return torch.cat((head, masks), dim=1).bool()

    def _masked_infonce_loss(self, x, y, mask=None):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        bsz, seqlen, dim = x.shape
        x_flat = x.reshape(bsz * seqlen, dim)
        y_flat = y.reshape(bsz * seqlen, dim)

        logits = torch.matmul(x_flat, y_flat.transpose(0, 1)) / self.tau
        targets = torch.arange(bsz * seqlen, device=x.device)

        if mask is not None:
            valid = mask.reshape(-1).bool()
            if not valid.any():
                return torch.zeros((), device=x.device)
            logits = logits.masked_fill(~valid.unsqueeze(0), float("-inf"))
            logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
            logits[targets, targets] = torch.where(valid, logits[targets, targets], torch.zeros_like(logits[targets, targets]))
            valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
            logits = logits[valid_idx]
            targets = targets[valid_idx]

        return F.cross_entropy(logits, targets)


    def calc_alignment_losses(self, pre_output, q_embed_data, masks=None, sem_pre_output=None, sem_q_embed_data=None, sem_masks=None):
        n_step = self.randomized_iteration_sampler() if self.same_step else None

        _, _, all_states_1 = self.recurrent_forward(
            pre_output,
            q_embed_data,
            num_steps=n_step,
            return_all_states=True,
        )

        if sem_pre_output is not None and sem_q_embed_data is not None:
            _, _, all_states_2 = self.recurrent_forward(
                sem_pre_output,
                sem_q_embed_data,
                num_steps=n_step if self.same_step else None,
                return_all_states=True,
            )
        else:
            _, _, all_states_2 = self.recurrent_forward(
                pre_output,
                q_embed_data,
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

    def _build_input_triplet(self, qseqs, cseqs, rseqs, shft_qseqs, shft_cseqs, shft_rseqs):
        pid_data = torch.cat((qseqs[:, 0:1], shft_qseqs), dim=1)
        q_data = torch.cat((cseqs[:, 0:1], shft_cseqs), dim=1)
        target = torch.cat((rseqs[:, 0:1], shft_rseqs), dim=1)
        return pid_data, q_data, target

    def _embed_inputs(self, pid_data, q_data, target):
        if self.emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        else:
            raise ValueError(f"Unsupported emb_type for lareskt: {self.emb_type}")

        if self.n_pid > 0 and self.emb_type.find("norasch") == -1:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            if self.emb_type.find("aktrasch") != -1:
                qa_embed_diff_data = self.qa_embed_diff(target)
                qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)

        return q_embed_data, qa_embed_data

    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()

        pid_data, q_data, target = self._build_input_triplet(q, c, r, qshft, cshft, rshft)
        q_embed_data, qa_embed_data = self._embed_inputs(pid_data, q_data, target)

        pre_output = self.pre_block(q_embed_data, qa_embed_data)
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
                sem_pre_output = self.pre_block(sem_q_embed_data, sem_qa_embed_data)

            tla_loss, sla_loss = self.calc_alignment_losses(
                pre_output,
                q_embed_data,
                masks=dcur.get("masks"),
                sem_pre_output=sem_pre_output,
                sem_q_embed_data=sem_q_embed_data if sem_pre_output is not None else None,
                sem_masks=sem_masks,
            )
            return preds, tla_loss, sla_loss

        if qtest:
            return preds, concat_q
        return preds
