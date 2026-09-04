#!/usr/bin/env python3
"""
Sweep the three knobs that decide whether this chip fits.

  --pix-thresh   how aggressively pixels binarise. Too low and every glyph
                 becomes a blob; too high and thin strokes vanish.
  --thresh       ternary quantisation threshold. Higher means more zero
                 weights, which means less silicon.
  features       how wide the frozen backbone is.

Stage 1 finds the best pixel threshold cheaply. Stage 2 sweeps sparsity
against feature width at that setting, which is the grid that matters.

Results append to sweep_results.csv as they finish, so you can watch it.
Weights land in weights/ for the area sweep to read.

    python3 sweep.py                     # full run, roughly an hour
    python3 sweep.py --quick             # fewer epochs, ~15 min
    tail -f sweep_results.csv            # in another terminal
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import train as T


def evaluate(
    xtr,
    ytr,
    xte,
    yte,
    n_feat,
    n_class,
    thresh,
    lr,
    epochs,
    batch,
    l1=0.0,
    seed=0,
    quiet=True,
):
    net = T.Net(n_feat, n_class, thresh=thresh, seed=seed)
    for k in net.opt:
        net.opt[k].lr = lr

    rng = np.random.default_rng(seed)
    n = len(xtr)
    best = 0.0
    for ep in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            idx = order[i : i + batch]
            net.backward(net.forward(xtr[idx]), ytr[idx], l1=l1)
        if ep >= epochs - 5 or ep % 10 == 0:
            acc = net.accuracy(xte, yte)
            best = max(best, acc)
            if not quiet:
                print(f"      epoch {ep:>3}  {acc:.4f}")

    W1q, b1, W2q, b2 = net.quantised()
    return {
        "accuracy": best,
        "zeros_W1": float((W1q == 0).mean()),
        "zeros_W2": float((W2q == 0).mean()),
    }, (W1q, b1, W2q, b2)


def save_weights(path, W1q, b1, W2q, b2, **meta):
    np.savez(
        path,
        W1=W1q.T.astype(np.int8),
        b1=np.round(b1).astype(np.int32),
        W2=W2q.T.astype(np.int8),
        b2=np.round(b2).astype(np.int32),
        **meta,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["emnist", "digits"], default="emnist")
    ap.add_argument("--split", default="balanced")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument(
        "--quick", action="store_true", help="fewer epochs and a smaller grid"
    )
    ap.add_argument("--out", default="sweep_results.csv")
    ap.add_argument("--weights-dir", default="weights")
    args = ap.parse_args()

    if args.quick:
        args.epochs = 12
        pix_grid = [0.25, 0.4]
        thresh_grid = [0.7, 1.1]
        feat_grid = [32, 48]
    else:
        pix_grid = [0.15, 0.25, 0.35, 0.45]
        thresh_grid = [0.7, 0.9, 1.1, 1.3]
        feat_grid = [32, 40, 48, 56]

    Path(args.weights_dir).mkdir(exist_ok=True)
    fields = [
        "stage",
        "pix_thresh",
        "ternary_thresh",
        "features",
        "accuracy",
        "zeros_W1",
        "zeros_W2",
        "seconds",
    ]
    with open(args.out, "w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()

    def record(row):
        with open(args.out, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)

    def load(pix):
        if args.dataset == "digits":
            xtr, xte, ytr, yte = T.load_digits_8x8()
            return xtr, ytr, xte, yte
        return T.load_emnist_hf(args.split, True, pix)

    # ---------------------------------------------------------- stage 1
    print("Stage 1 -- pixel threshold\n")
    print(f"{'pix':>6} {'ink':>6} {'accuracy':>9} {'zeros':>7} {'sec':>6}")
    print("-" * 40)

    stage1 = []
    for pix in pix_grid:
        xtr, ytr, xte, yte = load(pix)
        n_class = int(ytr.max()) + 1
        ink = float(xtr.mean())
        t0 = time.time()
        res, _ = evaluate(
            xtr, ytr, xte, yte, 48, n_class, 0.7, args.lr, args.epochs, args.batch
        )
        dt = time.time() - t0
        stage1.append((res["accuracy"], pix))
        print(
            f"{pix:>6.2f} {ink:>5.0%} {res['accuracy']:>9.4f} "
            f"{res['zeros_W1']:>6.1%} {dt:>6.0f}"
        )
        record(
            {
                "stage": 1,
                "pix_thresh": pix,
                "ternary_thresh": 0.7,
                "features": 48,
                "accuracy": round(res["accuracy"], 4),
                "zeros_W1": round(res["zeros_W1"], 4),
                "zeros_W2": round(res["zeros_W2"], 4),
                "seconds": round(dt),
            }
        )

    best_pix = max(stage1)[1]
    print(f"\nbest pixel threshold: {best_pix}\n")

    # ---------------------------------------------------------- stage 2
    print("Stage 2 -- sparsity against feature width\n")
    xtr, ytr, xte, yte = load(best_pix)
    n_class = int(ytr.max()) + 1

    print(f"{'thresh':>7} {'feat':>5} {'accuracy':>9} {'zeros W1':>9} {'sec':>6}  file")
    print("-" * 60)

    for th in thresh_grid:
        for nf in feat_grid:
            t0 = time.time()
            res, w = evaluate(
                xtr, ytr, xte, yte, nf, n_class, th, args.lr, args.epochs, args.batch
            )
            dt = time.time() - t0

            name = f"{args.weights_dir}/w_p{best_pix}_t{th}_f{nf}.npz"
            save_weights(
                name,
                *w,
                features=nf,
                classes=n_class,
                accuracy=res["accuracy"],
                pix_thresh=best_pix,
                ternary_thresh=th,
                zero_rate_backbone=res["zeros_W1"],
            )

            print(
                f"{th:>7.1f} {nf:>5} {res['accuracy']:>9.4f} "
                f"{res['zeros_W1']:>8.1%} {dt:>6.0f}  {name}"
            )
            record(
                {
                    "stage": 2,
                    "pix_thresh": best_pix,
                    "ternary_thresh": th,
                    "features": nf,
                    "accuracy": round(res["accuracy"], 4),
                    "zeros_W1": round(res["zeros_W1"], 4),
                    "zeros_W2": round(res["zeros_W2"], 4),
                    "seconds": round(dt),
                }
            )

    print(f"""
Done. Results in {args.out}, weights in {args.weights_dir}/

What to look for:
  - the highest zero rate you can reach without losing much accuracy.
    Sparsity is silicon; a point of accuracy may be worth several tiles.
  - whether wider features still help once sparsity is high.

Next: feed these .npz files into the area sweep instead of random weights.
""")


if __name__ == "__main__":
    main()
