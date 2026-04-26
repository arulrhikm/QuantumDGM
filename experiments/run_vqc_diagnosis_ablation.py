from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from experiments.bq_mps_baseline import fidelity_from_probs, find_chi_matching_vqc_params, mps_approximate_probs
from experiments.bq_oracle import evaluate_trained_circuit_bq, get_exact_probs_bq
from experiments.graph_families import erdos_renyi, sample_mrf_params
from src.entanglement import build_variational_circuit, get_entanglement_pairs
from src.train import kl_loss
from src.backends import BackendAdapter


def _init_params(mode: str, n_params: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if mode == "small_random":
        return rng.uniform(-0.1, 0.1, n_params)
    if mode == "wide_random":
        return rng.uniform(-0.5, 0.5, n_params)
    if mode == "zeros":
        return np.zeros(n_params, dtype=float)
    raise ValueError(f"Unknown init mode: {mode}")


def _train_with_optimizer(
    circuit,
    param_vec,
    target_probs: np.ndarray,
    init_params: np.ndarray,
    n_iters: int,
    lr: float,
    optimizer: str,
) -> tuple[np.ndarray, list[float]]:
    params = init_params.copy()
    history: list[float] = []
    shift = np.pi / 2.0
    backend = BackendAdapter()
    velocity = np.zeros_like(params)
    beta = 0.9

    for _ in range(n_iters):
        loss = kl_loss(circuit, param_vec, params, target_probs, backend, False)
        history.append(float(loss))
        grads = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += shift
            p_minus[i] -= shift
            l_plus = kl_loss(circuit, param_vec, p_plus, target_probs, backend, False)
            l_minus = kl_loss(circuit, param_vec, p_minus, target_probs, backend, False)
            grads[i] = 0.5 * (l_plus - l_minus)

        if optimizer == "sgd":
            params -= lr * grads
        elif optimizer == "momentum":
            velocity = beta * velocity + (1.0 - beta) * grads
            params -= lr * velocity
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    return params, history


def run(n: int, seeds: int, depths: list[int], optimizers: list[str], inits: list[str], iters: int):
    rows: list[dict] = []
    for seed in range(seeds):
        g = erdos_renyi(n, p=0.5, seed=seed)
        mrf_params = sample_mrf_params(g, seed=seed)
        target_probs, target_backend = get_exact_probs_bq(n, mrf_params, g)
        for depth in depths:
            pairs = get_entanglement_pairs("clique", n, g)
            circ, pvec = build_variational_circuit(n, depth, pairs)
            n_params = len(pvec)
            chi = find_chi_matching_vqc_params(n_params, n)

            t_mps = time.perf_counter()
            mps = mps_approximate_probs(n, mrf_params, g, chi_max=chi, n_shots=40_000)
            t_mps_s = time.perf_counter() - t_mps
            f_mps = fidelity_from_probs(mps["probs"], target_probs)

            for opt in optimizers:
                for init in inits:
                    init_params = _init_params(init, n_params, seed + 1000 * depth)
                    t0 = time.perf_counter()
                    trained, loss_hist = _train_with_optimizer(
                        circ, pvec, target_probs, init_params, iters, 0.03 if opt == "sgd" else 0.05, opt
                    )
                    t_vqc = time.perf_counter() - t0
                    bound = circ.assign_parameters(dict(zip(pvec, trained)))
                    vqc = evaluate_trained_circuit_bq(bound, target_probs)
                    rows.append(
                        {
                            "n": n,
                            "seed": seed,
                            "depth": depth,
                            "optimizer": opt,
                            "init": init,
                            "n_vqc_params": n_params,
                            "chi_max": chi,
                            "F_vqc": float(vqc["fidelity"]),
                            "F_mps": float(f_mps),
                            "gap_vqc_minus_mps": float(vqc["fidelity"] - f_mps),
                            "kl_vqc": float(vqc["kl"]),
                            "t_vqc_s": float(t_vqc),
                            "t_mps_s": float(t_mps_s),
                            "target_backend": target_backend,
                            "vqc_eval_backend": vqc["backend_used"],
                            "mps_backend": mps["backend_used"],
                            "loss_start": float(loss_hist[0]) if loss_hist else None,
                            "loss_end": float(loss_hist[-1]) if loss_hist else None,
                        }
                    )
                    print(
                        f"seed={seed} depth={depth} opt={opt} init={init} "
                        f"Fvqc={vqc['fidelity']:.3f} Fmps={f_mps:.3f}"
                    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--depths", default="3,4,5")
    parser.add_argument("--optimizers", default="sgd,momentum")
    parser.add_argument("--inits", default="small_random,wide_random,zeros")
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    rows = run(
        n=args.n,
        seeds=args.seeds,
        depths=[int(x) for x in args.depths.split(",") if x.strip()],
        optimizers=[x.strip() for x in args.optimizers.split(",") if x.strip()],
        inits=[x.strip() for x in args.inits.split(",") if x.strip()],
        iters=args.iters,
    )
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "vqc_ablation_n12.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
