#!/usr/bin/env python3
"""§53 cheap EAGLE-head acceptance probe (Track C endgame).

Modes:
  dry_run     — budget/ceiling table only (no GPU model load)
  linear_fit  — ridge-fit h_{t-1}→teacher logits on tip dumps; TF α/E
  smoke_head  — CE smoke on EagleDraftHead; TF + mandatory in-loop E
  in_loop     — load ckpt; mandatory in-loop E + timed c_d (no train)

Gates (§53):
  held-out E ≥ 2.4 before runtime wire
  runtime / in-loop E ≥ 2.2
  c_d ∈ 0.05–0.1× tip step restoring ceiling ≥420 at E=2.4

§52 lesson (binding): TF tip-dump E ≠ in-loop E. Never promote on TF alone.

Ledger §51 is reclaimed for verify kernels — this lever is §53.
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
        GATE_HELD_OUT_E,
        GATE_RUNTIME_E,
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
        "GATE_HELD_OUT_E": GATE_HELD_OUT_E,
        "GATE_RUNTIME_E": GATE_RUNTIME_E,
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
        reject_sample_prefix,
    )

    return acceptance_alpha, apply_temp_top_p, expected_accepted, reject_sample_prefix


def _ridge_fit_linear(features, targets, *, ridge: float = 1e-2):
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
    acceptance_alpha, apply_temp_top_p, _e, _r = _import_rejection()
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
    _aa, _at, expected_accepted, _r = _import_rejection()
    mean_a = sum(alphas) / len(alphas) if alphas else 0.0
    return {
        "positions": len(alphas),
        "mean_alpha": mean_a,
        "E_accepted_per_step": expected_accepted(mean_a, gamma),
        "E_by_gamma": {
            str(g): expected_accepted(mean_a, g) for g in (3, 4, 5)
        },
    }


def _load_windows(pack_path: Path) -> list[dict[str, Any]]:
    torch, _F = _import_torch()
    payload = torch.load(pack_path, map_location="cpu")
    windows = payload.get("windows") or payload.get("packs") or []
    if not windows and "logits_teacher" in payload:
        windows = [payload]
    return windows


def _split_train_heldout(windows: list[dict[str, Any]]) -> tuple[list, list]:
    train, held = [], []
    for w in windows:
        role = str(w.get("role") or "")
        song = str(w.get("song_id") or "")
        if role == "held_out" or song in ("nube-negra", "ela-ke-leitada"):
            held.append(w)
        else:
            train.append(w)
    if not held:
        # Single-song pack: use last 25% windows as held-out proxy.
        n = len(windows)
        cut = max(1, n // 4)
        return windows[:-cut], windows[-cut:]
    return train, held


def _eval_in_loop(
    head,
    windows: list[dict[str, Any]],
    *,
    gamma: int,
    temperature: float,
    top_p: float,
    max_windows: int,
    stride: int,
    device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Autoregressive γ-draft from head features; α vs tip teacher logits.

    Starts from real teacher h_t, then rolls head features for γ steps
    (not teacher-forced hiddens). Mandatory §52/§53 in-loop metric.
    """
    torch, _F = _import_torch()
    acceptance_alpha, apply_temp_top_p, _e, reject_sample_prefix = _import_rejection()

    head.eval()
    alphas: list[float] = []
    accepted_counts: list[int] = []
    timed_ms: list[float] = []
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for w in windows[:max_windows]:
            ht = w["hidden_teacher"].float()
            lt = w["logits_teacher"].float()
            if ht.ndim == 3:
                ht = ht[0]
            if lt.ndim == 3:
                lt = lt[0]
            t_len = int(min(ht.shape[0], lt.shape[0]))
            if t_len < gamma + 2:
                continue
            for start in range(0, t_len - gamma - 1, max(int(stride), 1)):
                h = ht[start].to(device)
                draft_logits = []
                if use_cuda:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                for _ in range(gamma):
                    logits, h = head.forward_with_features(h)
                    draft_logits.append(logits.float())
                if use_cuda:
                    torch.cuda.synchronize()
                    timed_ms.append((time.perf_counter() - t0) * 1e3)
                dlog = torch.stack(draft_logits, dim=0).cpu()
                tlog = lt[start + 1 : start + 1 + gamma]
                finite = torch.isfinite(tlog).all(dim=-1) & torch.isfinite(dlog).all(
                    dim=-1
                )
                if int(finite.sum()) != gamma:
                    continue
                p = apply_temp_top_p(tlog, temperature, top_p)
                q = apply_temp_top_p(dlog, temperature, top_p)
                a = acceptance_alpha(p, q)
                alphas.extend(float(x) for x in a.tolist() if x == x)
                draft_ids = torch.multinomial(q, 1).squeeze(-1)
                n_acc, _resid = reject_sample_prefix(
                    p_probs=p, q_probs=q, draft_token_ids=draft_ids
                )
                accepted_counts.append(int(n_acc) + (1 if int(n_acc) == gamma else 0))
    if not alphas:
        raise SystemExit("in-loop produced no finite alphas; check hidden pack")
    summary = _summarize(alphas, gamma)
    emp_e = sum(accepted_counts) / len(accepted_counts) if accepted_counts else 0.0
    timed_ms_sorted = sorted(timed_ms)
    c_d_ms = timed_ms_sorted[len(timed_ms_sorted) // 2] if timed_ms_sorted else None
    return summary, {
        "chains": len(accepted_counts),
        "empirical_E_accepted_per_step": emp_e,
        "closed_form_E_from_mean_alpha": summary["E_accepted_per_step"],
        "c_draft_ms_median": c_d_ms,
        "c_draft_ms_p90": timed_ms_sorted[int(0.9 * (len(timed_ms_sorted) - 1))]
        if timed_ms_sorted
        else None,
        "timed_chains": len(timed_ms),
        "stride": int(stride),
        "gamma": int(gamma),
        "note": "in-loop uses head-rolled features; teacher logits from tip dumps for p(x)",
    }


def _make_gate(
    eg: dict,
    *,
    teacher_force_E: float,
    held_out_E: float | None,
    in_loop_E: float | None,
    c_draft_ms: float,
) -> Any:
    ceil = eg["ceiling_tps"](
        c_draft_ms=c_draft_ms, e_acc=eg["GATE_HELD_OUT_E"]
    )
    return eg["EagleProbeGate"](
        teacher_force_E=float(teacher_force_E),
        held_out_E=held_out_E,
        in_loop_E=in_loop_E,
        c_draft_ms_est=float(c_draft_ms),
        ceiling_at_held_out=float(ceil),
    )


def _gate_dict(gate) -> dict[str, Any]:
    return {
        "decision": gate.decision(),
        "teacher_force_E": gate.teacher_force_E,
        "held_out_E": gate.held_out_E,
        "in_loop_E": gate.in_loop_E,
        "ceiling_at_held_out": gate.ceiling_at_held_out,
        "c_draft_ms_est": gate.c_draft_ms_est,
        "bars": {"held_out_E": 2.4, "runtime_E": 2.2, "ceiling_tps": 420.0},
    }


def run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    tip_step_ms = 1.85
    c_verify = 3.075
    gate_held = 2.4
    gate_rt = 2.2
    c_lo, c_hi = 0.05 * tip_step_ms, 0.10 * tip_step_ms
    d_model, vocab, hidden_mult, gamma = 768, 4097, 2, int(args.gamma)
    hid = d_model * hidden_mult
    flops_tok = 2.0 * (d_model * hid + hid * d_model + d_model * vocab)
    flops_chain = flops_tok * float(gamma)
    ms_chain_est = flops_chain / (10.0 * 1e12) * 1e3
    rows = []
    for frac in (0.05, 0.10):
        c_d = frac * tip_step_ms
        for e in (2.2, 2.4, 2.8):
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
    ceiling_e24 = 1000.0 * gate_held / (c_hi + c_verify)
    return {
        "schema": "s53-eagle-probe-dry-v1",
        "mode": "dry_run",
        "section": 53,
        "tip_commit": "55949274",
        "auth_fp16_main_tps": 366.11,
        "gates": {
            "held_out_E": gate_held,
            "runtime_E": gate_rt,
            "ceiling_tps": 420.0,
            "c_draft_ms_lo": c_lo,
            "c_draft_ms_hi": c_hi,
            "ceiling_at_E24_c_hi": ceiling_e24,
        },
        "flops_est": {
            "flops_per_token": flops_tok,
            "flops_gamma_chain": flops_chain,
            "ms_chain_est": ms_chain_est,
            "ms_per_token_est": ms_chain_est / float(gamma),
            "peak_tflops_assumed": 10.0,
            "gamma": float(gamma),
        },
        "budget_rows": rows,
        "decision": "GO_DUMP_THEN_PROBE",
        "note": (
            "dry_run only. Next: dump hiddens → linear_fit/smoke_head → in-loop. "
            "§51 reclaim: verify kernels. This lever is §53. "
            "§52: TF tip E≠in-loop E; do not promote on TF alone."
        ),
        "section52_e_collapse": {
            "runtime_E_median": 1.0,
            "runtime_E_map_weighted": 1.25,
            "offline_tf_E_1layer_g3": 1.97,
            "primary": "tf_tip_dump_vs_inloop_model_prefix_gap",
            "secondary": "draft_chain_no_top_p_no_processors",
        },
    }


def run_linear_fit_from_pack(args: argparse.Namespace) -> dict[str, Any]:
    torch, _F = _import_torch()
    eg = _import_eagle()
    pack_path = Path(args.hidden_pack)
    windows = _load_windows(pack_path)
    train_w, held_w = _split_train_heldout(windows)
    pool = train_w or windows
    all_f, all_y = [], []
    for w in pool:
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
        all_f.append(feats)
        all_y.append(lt[sl])
    if not all_f:
        raise SystemExit(
            f"no hidden_teacher windows in {pack_path}; regenerate dumps"
        )
    Fmat = torch.cat(all_f, dim=0)
    Ymat = torch.cat(all_y, dim=0)
    idx = torch.randperm(Fmat.shape[0])[: min(Fmat.shape[0], args.max_fit_rows)]
    W = _ridge_fit_linear(Fmat[idx], Ymat[idx], ridge=args.ridge)
    train_alphas = _alphas_from_logits(
        Ymat, Fmat @ W, temperature=args.temperature, top_p=args.top_p
    )
    held_alphas: list[float] = []
    for w in held_w:
        lt = w["logits_teacher"].float()
        ht = w["hidden_teacher"].float()
        if ht.ndim == 3:
            ht = ht[0]
        if lt.ndim == 3:
            lt = lt[0]
        feats, sl = eg["feature_shift_pairs"](ht)
        pred = feats @ W
        held_alphas.extend(
            _alphas_from_logits(
                lt[sl], pred, temperature=args.temperature, top_p=args.top_p
            )
        )
    summary = _summarize(train_alphas, args.gamma)
    held_summary = _summarize(held_alphas, args.gamma) if held_alphas else None
    flops = eg["estimate_mlp_head_ms"](gamma=args.gamma)
    c_d = max(float(flops["ms_chain_est"]), eg["C_DRAFT_TARGET_LO"])
    held_e = (
        float(held_summary["E_accepted_per_step"]) if held_summary else None
    )
    gate = _make_gate(
        eg,
        teacher_force_E=float(summary["E_accepted_per_step"]),
        held_out_E=held_e,
        in_loop_E=None,
        c_draft_ms=c_d,
    )
    return {
        "schema": "s53-eagle-probe-linear-v1",
        "mode": "linear_fit",
        "section": 53,
        "hidden_pack": str(pack_path),
        "windows_fit": len(pool),
        "n_train_windows": len(train_w),
        "n_held_windows": len(held_w),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "gamma": args.gamma,
        "acceptance_train_tf": summary,
        "acceptance_held_tf": held_summary,
        "flops_est": flops,
        "gate": _gate_dict(gate),
        "budget_rows": eg["budget_table"](),
        "note": "TF linear bound only; smoke_head/in_loop required before wire.",
    }


def run_smoke_head(args: argparse.Namespace) -> dict[str, Any]:
    torch, F = _import_torch()
    eg = _import_eagle()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = _load_windows(Path(args.hidden_pack))
    if not windows:
        raise SystemExit(f"empty windows in {args.hidden_pack}")
    train_w, held_w = _split_train_heldout(windows)
    train_pool = train_w or windows
    d_model = int(train_pool[0]["hidden_teacher"].shape[-1])
    vocab = int(train_pool[0]["logits_teacher"].shape[-1])
    head = eg["EagleDraftHead"](d_model=d_model, vocab_size=vocab).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    losses: list[float] = []
    t0 = time.perf_counter()
    head.train()
    for step in range(args.train_steps):
        w = train_pool[step % len(train_pool)]
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

    def _tf_on(ws):
        alphas: list[float] = []
        with torch.no_grad():
            for w in ws[: args.max_eval_windows]:
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
        return _summarize(alphas, args.gamma)

    tf_train = _tf_on(train_pool)
    tf_held = _tf_on(held_w) if held_w else None

    # Mandatory in-loop on held-out (fallback train if no held)
    loop_ws = held_w if held_w else train_pool
    in_loop_summary, in_loop_meta = _eval_in_loop(
        head,
        loop_ws,
        gamma=args.gamma,
        temperature=args.temperature,
        top_p=args.top_p,
        max_windows=args.max_eval_windows,
        stride=args.in_loop_stride,
        device=device,
    )
    # Also time/measure on train for reference
    train_loop_summary, train_loop_meta = _eval_in_loop(
        head,
        train_pool,
        gamma=args.gamma,
        temperature=args.temperature,
        top_p=args.top_p,
        max_windows=min(8, args.max_eval_windows),
        stride=max(args.in_loop_stride, 8),
        device=device,
    )

    ckpt = None
    if args.ckpt_out:
        ckpt = str(args.ckpt_out)
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "s53-eagle-head-smoke-v1",
                "state_dict": head.state_dict(),
                "d_model": d_model,
                "vocab_size": vocab,
                "train_steps": args.train_steps,
            },
            ckpt,
        )

    flops = eg["estimate_mlp_head_ms"](gamma=args.gamma)
    c_d_meas = in_loop_meta.get("c_draft_ms_median")
    c_d = float(c_d_meas) if c_d_meas is not None else max(
        float(flops["ms_chain_est"]), eg["C_DRAFT_TARGET_LO"]
    )
    held_e = float(in_loop_summary["E_accepted_per_step"])
    # Prefer empirical Leviathan E for runtime bar when available
    runtime_e = float(in_loop_meta["empirical_E_accepted_per_step"])
    gate = _make_gate(
        eg,
        teacher_force_E=float(
            (tf_held or tf_train)["E_accepted_per_step"]
        ),
        held_out_E=held_e,
        in_loop_E=runtime_e,
        c_draft_ms=c_d,
    )
    return {
        "schema": "s53-eagle-probe-smoke-v1",
        "mode": "smoke_head",
        "section": 53,
        "device": str(device),
        "train_steps": args.train_steps,
        "train_seconds": train_s,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "n_train_windows": len(train_pool),
        "n_held_windows": len(held_w),
        "acceptance_train_tf": tf_train,
        "acceptance_held_tf": tf_held,
        "acceptance_held_in_loop": in_loop_summary,
        "acceptance_train_in_loop": train_loop_summary,
        "in_loop_held": in_loop_meta,
        "in_loop_train": train_loop_meta,
        "ckpt_out": ckpt,
        "flops_est": flops,
        "gate": _gate_dict(gate),
        "budget_rows": eg["budget_table"](),
        "note": (
            "Held-out in-loop E is wire bar (≥2.4); empirical in-loop is runtime bar (≥2.2). "
            "§52: TF≠in-loop."
        ),
    }


def run_in_loop_only(args: argparse.Namespace) -> dict[str, Any]:
    torch, _F = _import_torch()
    eg = _import_eagle()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = _load_windows(Path(args.hidden_pack))
    train_w, held_w = _split_train_heldout(windows)
    ckpt = torch.load(args.ckpt_in, map_location="cpu")
    head = eg["EagleDraftHead"](
        d_model=int(ckpt["d_model"]), vocab_size=int(ckpt["vocab_size"])
    ).to(device)
    head.load_state_dict(ckpt["state_dict"])
    loop_ws = held_w if held_w else (train_w or windows)
    summary, meta = _eval_in_loop(
        head,
        loop_ws,
        gamma=args.gamma,
        temperature=args.temperature,
        top_p=args.top_p,
        max_windows=args.max_eval_windows,
        stride=args.in_loop_stride,
        device=device,
    )
    flops = eg["estimate_mlp_head_ms"](gamma=args.gamma)
    c_d_meas = meta.get("c_draft_ms_median")
    c_d = float(c_d_meas) if c_d_meas is not None else max(
        float(flops["ms_chain_est"]), eg["C_DRAFT_TARGET_LO"]
    )
    gate = _make_gate(
        eg,
        teacher_force_E=0.0,
        held_out_E=float(summary["E_accepted_per_step"]),
        in_loop_E=float(meta["empirical_E_accepted_per_step"]),
        c_draft_ms=c_d,
    )
    return {
        "schema": "s53-eagle-probe-inloop-v1",
        "mode": "in_loop",
        "section": 53,
        "ckpt_in": str(args.ckpt_in),
        "n_held_windows": len(held_w),
        "acceptance_held_in_loop": summary,
        "in_loop_held": meta,
        "flops_est": flops,
        "gate": _gate_dict(gate),
        "budget_rows": eg["budget_table"](),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("dry_run", "linear_fit", "smoke_head", "in_loop"),
        default="dry_run",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hidden-pack", type=Path, default=None)
    ap.add_argument("--ckpt-out", type=Path, default=None)
    ap.add_argument("--ckpt-in", type=Path, default=None)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--max-fit-rows", type=int, default=4096)
    ap.add_argument("--train-steps", type=int, default=400)
    ap.add_argument("--max-eval-windows", type=int, default=32)
    ap.add_argument("--in-loop-stride", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    if args.mode == "dry_run":
        result = run_dry_run(args)
    elif args.mode == "linear_fit":
        if args.hidden_pack is None:
            raise SystemExit("--hidden-pack required for linear_fit")
        result = run_linear_fit_from_pack(args)
    elif args.mode == "smoke_head":
        if args.hidden_pack is None:
            raise SystemExit("--hidden-pack required for smoke_head")
        result = run_smoke_head(args)
    else:
        if args.hidden_pack is None or args.ckpt_in is None:
            raise SystemExit("--hidden-pack and --ckpt-in required for in_loop")
        result = run_in_loop_only(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result.get("gate") or {"decision": result.get("decision")}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
