"""Masked local eligibility gradients and bounded core optimizer steps."""

import numpy as np
import torch


def assemble_gradient(signals, input_eligibility, recurrent_eligibility, rec_mask):
    """Symmetric e-prop: sum_t learning_signal_j(t) * filtered_eligibility_ij(t)."""
    gi = np.einsum("tn,tin->in", signals, input_eligibility)
    gr = np.einsum("tn,tin->in", signals, recurrent_eligibility) * rec_mask
    return gi, gr


def eprop_vector(signal, input_eligibility, recurrent_eligibility, mask):
    a, b = assemble_gradient(signal, input_eligibility, recurrent_eligibility, mask)
    return np.r_[a.ravel(), b.ravel()]


def apply_core(agent, optimizer, g, max_update=0.0):
    before = [p.detach().clone() for p in agent.rnn.parameters()] if max_update > 0 else None
    ni = agent.rnn.win.numel()
    agent.rnn.win.grad = torch.tensor(g[:ni].reshape(tuple(agent.rnn.win.shape))).to(agent.rnn.win)
    agent.rnn.wrec.grad = torch.tensor(g[ni:].reshape(tuple(agent.rnn.wrec.shape))).to(
        agent.rnn.wrec
    )
    torch.nn.utils.clip_grad_norm_(agent.rnn.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if before is not None:
        with torch.no_grad():
            norm = torch.sqrt(
                sum((p - old).square().sum() for p, old in zip(agent.rnn.parameters(), before))
            )
            factor = min(1.0, max_update / max(float(norm), 1e-30))
            for p, old in zip(agent.rnn.parameters(), before):
                p.copy_(old + factor * (p - old))
    with torch.no_grad():
        agent.rnn.wrec *= agent.rnn.rec_mask
