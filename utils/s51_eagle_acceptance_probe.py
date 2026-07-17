#!/usr/bin/env python3
"""§51 cheap EAGLE-head acceptance probe (before heavy train).

Modes:
  dry_run     — budget/ceiling table only (no GPU model load)
  linear_fit  — ridge-fit h_{t-1}→teacher logits on tip dumps; TF α/E
  smoke_head  — few hundred CE steps on EagleDraftHead; TF α/E

§52 lesson (binding): teacher-force tip-dump E overstated in-loop E (~2.0 → ~1.1).
This probe never authorizes runtime on TF alone. Optional ``--in-loop-windows``
runs a short turbo-style Leviathan loop once a head ckpt exists (later rung).

Not a 500 / TIER1 claim. Tip stays 55949274 / 366.11.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get("PROBE_REPO", ".")).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _import_torch():
    import torch
    import torch.nn.functional as F

    return torch, F


def _import_eagle():
    from osuT5.osuT5.inference.turbo.eagle_draft import (
        C_DRAFT_TARGET_HI,
        C_DRAFT_TARGET_LO,
        GATE_CEILING_TPS,
        GATE_E,
        EagleDraftHead,
        EagleProbeGate,
        budget_table,
        ceiling_tps,
        estimate_mlp_head_ms,
        feature_shift_pairs,
    )

    return {
        "C_DRAFT_TARGET_HI": C_DRAFT_TARGET_HI,
        "C_DRAFT_TARGET_LO": C_DRAFT_TARGET_LO,
        "GATE_CEILING_TPS": GATE_CEILING_TPS,
        "GATE_E": GATE_E,
        "EagleDraftHead": EagleDraftHead,
        "EagleProbeGate": EagleProbeGate,
        "budget_table": budget_table,
        "ceiling_tps": ceiling_tps,
        "estimate_mlp_head_ms": estimate_mlp_head_ms,
        "feature_shift_pairs": feature_shift_pairs,
    }


def _import_rejection():
    from osuT5.osuT5.inference.turbo.rejection import (
        acceptance_alpha,
        apply_temp_top_p,
        expected_accepted,
    )

    return acceptance_alpha, apply_temp_top_p, expected_accepted


def _ridge_fit_linear(features, targets, *, ridge: float = 1e-2):
    """Solve min ||F W - Y||_F^2 + ridge ||W||^2; F[N,D], Y[N,V] → W[D,V]."""
    torch, _F = _import_torch()
    f = features.float()
    y = targets.float()
    d = f.shape[1]
    ft_f = f.T @ f
    ft_f = ft_f + ridge * torch.eye(d, device=f.device, dtype=f.dtype)
    ft_y = f.T @ y
    return torch.linalg.solve(ft_f, ft_y)


def _alphas_from_logits(
    teacher_logits,
    draft_logits,
    *,
    temperature: float,
    top_p: float,
) -> list[float]:
    torch, _F = _import_torch()
    acceptance_alpha, apply_temp_top_p, _expected = _import_rejection()
    finite = torch.isfinite(teacher_logits).all(dim=-1) & torch.isfinite(
        draft_logits
    ).all(dim=-1)
    if int(finite.sum()) == 0:
        return []
    p = apply_temp_top_p(teacher_logits[finite], temperature, top_p)
    q = apply_temp_top_p(draft_logits[finite], temperature, top_p)
    a = acceptance_alpha(p, q)
    return [float(x) for x in a.tolist() if x == x]


def _summarize(alphas: list[float], gamma: int) -> dict[str, Any]:
    _aa, _at, expected_accepted = _import_rejection()
    mean_a = sum(alphas) / len(alphas) if alphas else 0.0
    return {
        "positions": len(alphas),
        "mean_alpha": mean_a,
        "E_accepted_per_step": expected_accepted(mean_a, gamma),
        "E_by_gamma": {
            str(g): expected_accepted(mean_a, g) for g in (3, 4, 5)
        },
    }


def run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    # Torch-free path so login nodes without the env can still emit the plan.
    tip_step_ms = 1.85
    c_verify = 3.075
    gate_e = 1.7
    c_lo, c_hi = 0.05 * tip_step_ms, 0.10 * tip_step_ms
    d_model, vocab, hidden_mult, gamma = 768, 4097, 2, int(args.gamma)
    hid = d_model * hidden_mult
    flops_tok = 2.0 * (d_model * hid + hid * d_model + d_model * vocab)
    flops_chain = flops_tok * float(gamma)
    ms_chain_est = flops_chain / (10.0 * 1e12) * 1e3
    flops = {
        "flops_per_token": flops_tok,
        "flops_gamma_chain": flops_chain,
        "ms_chain_est": ms_chain_est,
        "ms_per_token_est": ms_chain_est / float(gamma),
        "peak_tflops_assumed": 10.0,
        "gamma": float(gamma),
    }
    rows = []
    for frac in (0.05, 0.10):
        c_d = frac * tip_step_ms
        for e in (1.7, 2.0, 2.4):
            step = c_d + c_verify
            rows.append(
                {
                    "c_d_frac": frac,
                    "c_draft_ms": c_d,
                    "E": float(e),
                    "step_ms": step,
                    "ceiling_tps": 1000.0 * float(e) / step,
                }
            )
    ceiling_e17 = 1000.0 * gate_e / (c_hi + c_verify)
    decision = "GO_SMOKE_TRAIN"  # dry_run placeholder; real gate needs TF/in-loop E
    return {
        "schema": "s51-eagle-probe-dry-v1",
        "mode": "dry_run",
        "tip_commit": "55949274",
        "auth_fp16_main_tps": 366.11,
        "gates": {
            "E": gate_e,
            "ceiling_tps": 420.0,
            "c_draft_ms_lo": c_lo,
            "c_draft_ms_hi": c_hi,
            "ceiling_at_E17_c_hi": ceiling_e17,
        },
        "flops_est": flops,
        "budget_rows": rows,
        "decision": decision,
        "note": (
            "dry_run only — no weights. Next: linear_fit / smoke_head on GPU. "
            "§52: TF tip E≠in-loop E; do not promote on TF alone."
        ),
        "section52_e_collapse": {
            "runtime_E_median": 1.0,
            "runtime_E_map_weighted": 1.25,
            "offline_tf_E_1layer_g3": 1.97,
            "primary": "tf_tip_dump_vs_inloop_model_prefix_gap",
            "secondary": "draft_chain_no_top_p_no_processors",
            "not_easy_17_fix": True,
        },
    }


def run_linear_fit_from_pack(args: argparse.Namespace) -> dict[str, Any]:
    """Ridge-fit h_{t-1}→teacher logits. Pack must carry hidden_teacher."""
    torch, _F = _import_torch()
    eg = _import_eagle()
    pack_path = Path(args.hidden_pack)
    payload = torch.load(pack_path, map_location="cpu")
    windows = payload.get("windows") or payload.get("packs") or []
    if not windows and "logits_teacher" in payload:
        windows = [payload]
    alphas: list[float] = []
    n_fit = 0
    for w in windows:
        lt = w.get("logits_teacher")
        ht = w.get("hidden_teacher") or w.get("decoder_hidden")
        if lt is None or ht is None:
            continue
        lt = lt.float()
        ht = ht.float()
        if ht.ndim == 3:
            ht = ht[0]
        if lt.ndim == 3:
            lt = lt[0]
        feats, sl = eg["feature_shift_pairs"](ht)
        targets = lt[sl]
        n = int(feats.shape[0])
        if n < 8:
            continue
        idx = torch.randperm(n)[: min(n, args.max_fit_rows)]
        w_mat = _ridge_fit_linear(feats[idx], targets[idx], ridge=args.ridge)
        pred = feats @ w_mat
        alphas.extend(
            _alphas_from_logits(
                targets, pred, temperature=args.temperature, top_p=args.top_p
            )
        )
        n_fit += 1
    if not alphas:
        raise SystemExit(
            f"no hidden_teacher windows in {pack_path}; "
            "regenerate dumps with output_hidden_states or use smoke_head"
        )
    summary = _summarize(alphas, args.gamma)
    flops = eg["estimate_mlp_head_ms"](gamma=args.gamma)
    c_d = max(float(flops["ms_chain_est"]), eg["C_DRAFT_TARGET_LO"])
    gate = eg["EagleProbeGate"](
        teacher_force_E=float(summary["E_accepted_per_step"]),
        in_loop_E=None,
        c_draft_ms_est=c_d,
        ceiling_at_E17=eg["ceiling_tps"](c_draft_ms=c_d, e_acc=eg["GATE_E"]),
    )
    return {
        "schema": "s51-eagle-probe-linear-v1",
        "mode": "linear_fit",
        "hidden_pack": str(pack_path),
        "windows_fit": n_fit,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "gamma": args.gamma,
        "acceptance": summary,
        "flops_est": flops,
        "gate": {
            "decision": gate.decision(),
            "teacher_force_E": gate.teacher_force_E,
            "ceiling_at_E17": gate.ceiling_at_E17,
            "c_draft_ms_est": gate.c_draft_ms_est,
        },
        "budget_rows": eg["budget_table"](),
        "note": "TF-only linear upper bound; still need in-loop E≥1.7 before runtime wire.",
    }


def run_smoke_head(args: argparse.Namespace) -> dict[str, Any]:
    """Short CE smoke on EagleDraftHead using teacher hidden dumps."""
    torch, F = _import_torch()
    eg = _import_eagle()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack_path = Path(args.hidden_pack)
    payload = torch.load(pack_path, map_location="cpu")
    windows = payload.get("windows") or []
    if not windows:
        raise SystemExit(f"empty windows in {pack_path}")
    d_model = int(windows[0]["hidden_teacher"].shape[-1])
    vocab = int(windows[0]["logits_teacher"].shape[-1])
    head = eg["EagleDraftHead"](d_model=d_model, vocab_size=vocab).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    losses: list[float] = []
    t0 = time.perf_counter()
    head.train()
    for step in range(args.train_steps):
        w = windows[step % len(windows)]
        ht = w["hidden_teacher"].float()
        lt = w["logits_teacher"].float()
        if ht.ndim == 3:
            ht = ht[0]
        if lt.ndim == 3:
            lt = lt[0]
        feats, sl = eg["feature_shift_pairs"](ht)
        labels = lt[sl].argmax(dim=-1)
        feats = feats.to(device)
        labels = labels.to(device)
        opt.zero_grad(set_to_none=True)
        logits = head(feats)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    train_s = time.perf_counter() - t0
    head.eval()
    alphas: list[float] = []
    with torch.no_grad():
        for w in windows[: args.max_eval_windows]:
            ht = w["hidden_teacher"].float()
            lt = w["logits_teacher"].float()
            if ht.ndim == 3:
                ht = ht[0]
            if lt.ndim == 3:
                lt = lt[0]
            feats, sl = eg["feature_shift_pairs"](ht)
            pred = head(feats.to(device)).float().cpu()
            alphas.extend(
                _alphas_from_logits(
                    lt[sl], pred, temperature=args.temperature, top_p=args.top_p
                )
            )
    summary = _summarize(alphas, args.gamma)
    ckpt = None
    if args.ckpt_out:
        ckpt = str(args.ckpt_out)
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "s51-eagle-head-smoke-v1",
                "state_dict": head.state_dict(),
                "d_model": d_model,
                "vocab_size": vocab,
                "train_steps": args.train_steps,
            },
            ckpt,
        )
    flops = eg["estimate_mlp_head_ms"](gamma=args.gamma)
    c_d = max(float(flops["ms_chain_est"]), eg["C_DRAFT_TARGET_LO"])
    gate = eg["EagleProbeGate"](
        teacher_force_E=float(summary["E_accepted_per_step"]),
        in_loop_E=None,
        c_draft_ms_est=c_d,
        ceiling_at_E17=eg["ceiling_tps"](c_draft_ms=c_d, e_acc=eg["GATE_E"]),
    )
    return {
        "schema": "s51-eagle-probe-smoke-v1",
        "mode": "smoke_head",
        "device": str(device),
        "train_steps": args.train_steps,
        "train_seconds": train_s,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "acceptance": summary,
        "ckpt_out": ckpt,
        "flops_est": flops,
        "gate": {
            "decision": gate.decision(),
            "teacher_force_E": gate.teacher_force_E,
            "ceiling_at_E17": gate.ceiling_at_E17,
            "c_draft_ms_est": gate.c_draft_ms_est,
        },
        "budget_rows": eg["budget_table"](),
        "note": "Smoke TF gate only; in-loop E≥1.7 still required (§52).",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("dry_run", "linear_fit", "smoke_head"),
        default="dry_run",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hidden-pack", type=Path, default=None)
    ap.add_argument("--ckpt-out", type=Path, default=None)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--max-fit-rows", type=int, default=4096)
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--max-eval-windows", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    if args.mode == "dry_run":
        result = run_dry_run(args)
    elif args.mode == "linear_fit":
        if args.hidden_pack is None:
            raise SystemExit("--hidden-pack required for linear_fit")
        result = run_linear_fit_from_pack(args)
    else:
        if args.hidden_pack is None:
            raise SystemExit("--hidden-pack required for smoke_head")
        result = run_smoke_head(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result.get("gate") or {"decision": result.get("decision")}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
