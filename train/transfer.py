#!/usr/bin/env python3
"""
The go/no-go experiment.

Train a backbone on many classes, freeze it, then fit a fresh 6-class head on
character sets the backbone was never trained on. Compare against training the
whole network on those 6 classes, which is the ceiling.

The gap between the two is the cost of freezing. That number is the thesis.

    pass  mean gap under 5 points, worst set under 8
    fail  the backbone is not general enough; fix it before writing RTL

    python3 transfer.py                       # 20 random 6-class sets
    python3 transfer.py --features 32 48      # compare widths
    python3 transfer.py --holdout 12          # hold out more classes
"""

import argparse
import time

import numpy as np

import train as T


def fit_head(net, xtr, ytr, xte, yte, n_class, epochs, batch, lr, seed=0):
    """Train only the head. The backbone weights never move."""
    rng = np.random.default_rng(seed)
    head = T.Net(net.W1.shape[1], n_class, thresh=net.thresh, seed=seed)
    head.W1 = net.W1.copy()  # frozen
    head.b1 = net.b1.copy()
    for k in head.opt:
        head.opt[k].lr = lr

    n = len(xtr)
    best = 0.0
    for ep in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            idx = order[i : i + batch]
            logits = head.forward(xtr[idx])
            # backward, but discard the backbone updates
            W1, b1 = head.W1.copy(), head.b1.copy()
            head.backward(logits, ytr[idx])
            head.W1, head.b1 = W1, b1
        if ep >= epochs - 3:
            best = max(best, head.accuracy(xte, yte))
    return best


def fit_full(xtr, ytr, xte, yte, n_feat, n_class, thresh, epochs, batch, lr, seed=0):
    """Train everything on these 6 classes. This is the ceiling."""
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
            net.backward(net.forward(xtr[idx]), ytr[idx])
        if ep >= epochs - 3:
            best = max(best, net.accuracy(xte, yte))
    return best


def subset(x, y, classes):
    """Pull out the rows for a set of classes and relabel them 0..k-1."""
    mask = np.isin(y, classes)
    remap = {c: i for i, c in enumerate(classes)}
    return x[mask], np.array([remap[v] for v in y[mask]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=int, nargs="+", default=[48])
    ap.add_argument("--head-classes", type=int, default=6)
    ap.add_argument(
        "--holdout",
        type=int,
        default=10,
        help="classes withheld from backbone training",
    )
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--pix-thresh", type=float, default=0.25)
    ap.add_argument("--thresh", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--backbone-epochs", type=int, default=200)
    ap.add_argument("--head-epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--split", default="balanced")
    args = ap.parse_args()

    print(f"loading emnist '{args.split}' ...")
    xtr, ytr, xte, yte = T.load_emnist_hf(args.split, True, args.pix_thresh)
    n_total = int(ytr.max()) + 1
    print(f"  {len(xtr)} train, {n_total} classes, ink {xtr.mean():.0%}\n")

    rng = np.random.default_rng(0)
    held = sorted(rng.choice(n_total, args.holdout, replace=False).tolist())
    trained_on = [c for c in range(n_total) if c not in held]
    print(f"backbone trains on {len(trained_on)} classes")
    print(f"heads are fitted on 6 drawn from the {len(held)} held out: {held}\n")

    bx, by = subset(xtr, ytr, trained_on)
    bxe, bye = subset(xte, yte, trained_on)

    for n_feat in args.features:
        print(f"=== {n_feat} features ===")
        t0 = time.time()
        backbone = T.Net(n_feat, len(trained_on), thresh=args.thresh, seed=0)
        for k in backbone.opt:
            backbone.opt[k].lr = args.lr
        r = np.random.default_rng(0)
        for ep in range(args.backbone_epochs):
            order = r.permutation(len(bx))
            for i in range(0, len(bx), args.batch):
                idx = order[i : i + args.batch]
                backbone.backward(backbone.forward(bx[idx]), by[idx])
        acc_bb = backbone.accuracy(bxe, bye)
        W1q, _, _, _ = backbone.quantised()
        print(
            f"  backbone: {acc_bb:.4f} on {len(trained_on)} classes, "
            f"{(W1q == 0).mean():.1%} zeros, {time.time() - t0:.0f}s\n"
        )

        print(f"  {'set':>26} {'frozen':>8} {'ceiling':>8} {'gap':>7}")
        print("  " + "-" * 52)

        gaps = []
        for t in range(args.trials):
            cls = sorted(
                np.random.default_rng(100 + t)
                .choice(held, args.head_classes, replace=False)
                .tolist()
            )
            hx, hy = subset(xtr, ytr, cls)
            hxe, hye = subset(xte, yte, cls)

            a_frozen = fit_head(
                backbone,
                hx,
                hy,
                hxe,
                hye,
                args.head_classes,
                args.head_epochs,
                args.batch,
                args.lr,
                seed=t,
            )
            a_ceil = fit_full(
                hx,
                hy,
                hxe,
                hye,
                n_feat,
                args.head_classes,
                args.thresh,
                args.head_epochs,
                args.batch,
                args.lr,
                seed=t,
            )
            gap = a_ceil - a_frozen
            gaps.append(gap)
            print(f"  {str(cls):>26} {a_frozen:>8.3f} {a_ceil:>8.3f} {gap:>+7.3f}")

        g = np.array(gaps)
        print(f"\n  mean gap  {g.mean():+.3f}")
        print(f"  worst     {g.max():+.3f}")
        print(f"  std       {g.std():.3f}")
        verdict = (
            "PASS"
            if g.mean() < 0.05 and g.max() < 0.08
            else "MARGINAL"
            if g.mean() < 0.08
            else "FAIL"
        )
        print(f"  verdict   {verdict}\n")

    print("""
Reading this:
  The gap is what freezing costs. Small gap means the frozen features are
  general enough that a new head can do its job, which is the whole claim.

  A low backbone accuracy is not itself a problem. What matters is whether
  the 6-class heads land somewhere useful and close to their ceiling.
""")


if __name__ == "__main__":
    main()
