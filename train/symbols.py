#!/usr/bin/env python3
"""
Fit a head on symbols that are NOT in EMNIST.

The backbone stays exactly as taped out -- 3,584 ternary weights trained on
handwritten characters, frozen, never retrained. Only the 336 head weights
are fitted. If this works, the frozen features are general enough to describe
shapes the silicon was never trained toward, which is the whole claim.

    python3 symbols.py                    fit the default six
    python3 symbols.py --show             print the templates
    python3 symbols.py --set faces        a different six
    python3 symbols.py --compare          symbols vs EMNIST letters

Writes a 90-byte bitstream you can shift into the real chip.
"""

import argparse
import json
from pathlib import Path

import chipsim
import numpy as np

HERE = Path(__file__).resolve().parent

# 8x8 templates. '#' is ink. Nothing here resembles a letter or digit.
TEMPLATES = {
    "smile": """
        ........
        .#....#.
        .#....#.
        ........
        #......#
        .#....#.
        ..####..
        ........""",
    "frown": """
        ........
        .#....#.
        .#....#.
        ........
        ..####..
        .#....#.
        #......#
        ........""",
    "heart": """
        ........
        .##..##.
        ########
        ########
        .######.
        ..####..
        ...##...
        ........""",
    "arrow_up": """
        ...##...
        ..####..
        .######.
        ########
        ...##...
        ...##...
        ...##...
        ...##...""",
    "arrow_rt": """
        ........
        ....#...
        ....##..
        ########
        ########
        ....##..
        ....#...
        ........""",
    "cross": """
        ........
        .##..##.
        ..####..
        ...##...
        ...##...
        ..####..
        .##..##.
        ........""",
    "check": """
        ........
        ......##
        .....##.
        #...##..
        ##.##...
        .####...
        ..##....
        ........""",
    "star": """
        ...##...
        ...##...
        ########
        .######.
        ..####..
        .##..##.
        ##....##
        ........""",
    "wave": """
        ........
        ........
        .##...##
        ##.##.##
        ....##..
        ........
        ........
        ........""",
    "box": """
        ........
        .######.
        .#....#.
        .#....#.
        .#....#.
        .#....#.
        .######.
        ........""",
    "circle": """
        ........
        ..####..
        .##..##.
        .#....#.
        .#....#.
        .##..##.
        ..####..
        ........""",
    "disc": """
        ........
        ..####..
        .######.
        .######.
        .######.
        .######.
        ..####..
        ........""",
    "triangle": """
        ........
        ...##...
        ...##...
        ..####..
        ..####..
        .######.
        ########
        ........""",
    "diamond": """
        ........
        ...##...
        ..####..
        .######.
        .######.
        ..####..
        ...##...
        ........""",
    "line_h": """
        ........
        ........
        ........
        ########
        ########
        ........
        ........
        ........""",
    "line_v": """
        ...##...
        ...##...
        ...##...
        ...##...
        ...##...
        ...##...
        ...##...
        ...##...""",
    "plus": """
        ........
        ...##...
        ...##...
        ########
        ########
        ...##...
        ...##...
        ........""",
    "slash": """
        ......##
        .....##.
        ....##..
        ...##...
        ..##....
        .##.....
        ##......
        ........""",
}

SETS = {
    "default": ["smile", "frown", "heart", "arrow_up", "cross", "check"],
    "faces":   ["smile", "frown", "heart", "star", "wave", "box"],
    "arrows":  ["arrow_up", "arrow_rt", "cross", "check", "box", "star"],
    "shapes":  ["circle", "box", "triangle", "diamond", "line_h", "line_v"],
    "shapes2": ["disc", "plus", "cross", "slash", "star", "line_h"],
}


def parse(t):
    rows = [r.strip() for r in t.strip().split("\n")]
    return np.array([[1 if c == "#" else 0 for c in r] for r in rows],
                    dtype=np.int32)


def show(name, g):
    print(f"  {name}")
    for row in g:
        print("    " + "".join("##" if v else ". " for v in row))


def augment(base, n, rng):
    """Make n noisy variants: shift, dilate/erode, salt-and-pepper.

    Roughly the variation you would get from someone drawing the same symbol
    repeatedly on a small grid.
    """
    out = np.zeros((n, 8, 8), dtype=np.int32)
    for k in range(n):
        g = base.copy()

        # shift by up to one cell
        dy, dx = rng.integers(-1, 2), rng.integers(-1, 2)
        g = np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        if dy > 0: g[:dy] = 0
        elif dy < 0: g[dy:] = 0
        if dx > 0: g[:, :dx] = 0
        elif dx < 0: g[:, dx:] = 0

        # stroke weight
        r = rng.random()
        if r < 0.25:                                   # thicken
            p = np.pad(g, 1)
            g = np.maximum.reduce([p[:-2, 1:-1], p[2:, 1:-1],
                                   p[1:-1, :-2], p[1:-1, 2:], g])
        elif r < 0.4:                                  # thin
            p = np.pad(g, 1)
            g = np.minimum.reduce([p[:-2, 1:-1], p[2:, 1:-1], g])

        # speckle
        flip = rng.random((8, 8)) < 0.06
        g = np.where(flip, 1 - g, g)
        out[k] = g
    return out.reshape(n, 64)


def build(names, n_per=1500, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i, nm in enumerate(names):
        base = parse(TEMPLATES[nm])
        xs.append(augment(base, n_per, rng))
        ys.append(np.full(n_per, i))
    x, y = np.vstack(xs), np.concatenate(ys)
    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]
    cut = int(0.8 * len(x))
    return x[:cut], y[:cut], x[cut:], y[cut:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="default", choices=list(SETS))
    ap.add_argument("--symbols", nargs=6, help="pick six by name")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--weights", default=str(HERE / "weights_f56.npz"))
    ap.add_argument("--out", default="heads")
    ap.add_argument("--no-centre", action="store_true",
                    help="skip centre-of-mass normalisation (costs ~15 points)")
    args = ap.parse_args()

    names = args.symbols or SETS[args.set]
    for nm in names:
        if nm not in TEMPLATES:
            raise SystemExit(f"unknown symbol {nm!r}. "
                             f"available: {', '.join(TEMPLATES)}")

    if args.show:
        for nm in names:
            show(nm, parse(TEMPLATES[nm]))
        return

    d = np.load(args.weights)
    W1, b1 = d["W1"].astype(int), d["b1"].astype(int)
    print(f"backbone: {W1.shape[0]} features, {(W1 == 0).mean():.1%} zeros")
    print("          trained on EMNIST handwriting, FROZEN, never retrained")
    print(f"symbols:  {' '.join(names)}")
    print("          none of these appear in EMNIST\n")

    xtr, ytr, xte, yte = build(names)
    print(f"generated {len(xtr)} train / {len(xte)} test samples")

    if not args.no_centre:
        xtr, xte = chipsim.centre(xtr), chipsim.centre(xte)
        print("centred by centre of mass (EMNIST is centred, so the frozen")
        print("features are not translation invariant)")

    # the frozen half -- exactly what the silicon computes
    Ftr = ((xtr @ W1.T + b1) >= 0).astype(np.int32)
    Fte = ((xte @ W1.T + b1) >= 0).astype(np.int32)

    # how much do the frozen features even separate these?
    uniq = len(np.unique(Ftr, axis=0))
    print(f"distinct feature patterns: {uniq} of {len(Ftr)} samples\n")

    print("fitting head (336 weights + 6 biases) ...")
    W2, b2 = chipsim.fit_head(Ftr, ytr, epochs=args.epochs)
    pred = (Fte @ W2.T + b2).argmax(axis=1)
    acc = (pred == yte).mean()

    print(f"\n  accuracy on held-out symbols: {acc:.3f}\n")
    print(f"  {'symbol':>10}  {'accuracy':>8}")
    print("  " + "-" * 22)
    for i, nm in enumerate(names):
        m = yte == i
        print(f"  {nm:>10}  {(pred[m] == i).mean():>8.3f}")

    # confusion, only if something is going wrong
    if acc < 0.95:
        print("\n  confusion (row = true):")
        print("        " + " ".join(f"{n[:4]:>5}" for n in names))
        for i, nm in enumerate(names):
            row = [int(((yte == i) & (pred == j)).sum()) for j in range(6)]
            print(f"  {nm[:6]:>6} " + " ".join(f"{v:>5}" for v in row))

    stream = chipsim.pack_head(W2, b2)
    chip = chipsim.Chip(W1, b1)
    chip.load_head(stream)
    n = min(300, len(xte))
    sim = np.array([chip.classify(x) for x in xte[:n]])
    if not np.array_equal(sim, pred[:n]):
        print("\n  WARNING bitstream round-trip mismatch")

    Path(args.out).mkdir(exist_ok=True)
    path = Path(args.out) / f"head_symbols_{args.set}.json"
    path.write_text(json.dumps({
        "characters": [n[:5] for n in names],
        "symbol_names": names,
        "accuracy": float(acc),
        "note": "backbone frozen on EMNIST; these symbols are not in EMNIST",
        "bits": chipsim.HEAD_BITS,
        "bytes": chipsim.HEAD_BITS // 8,
        "bitstream": stream,
        "templates": {n: TEMPLATES[n].strip() for n in names},
    }, indent=2))
    print(f"\n  wrote {path}  ({chipsim.HEAD_BITS // 8} bytes)")

    if args.compare:
        print("\n--- for comparison, six EMNIST letters on the same backbone ---")
        try:
            xt, yt, xv, yv = chipsim.load_emnist()
            cls = [chipsim.BYCLASS.index(c) for c in "ABCDEF"]
            ftr, ltr = chipsim.subset(((xt @ W1.T + b1) >= 0).astype(np.int32),
                                      yt, cls)
            fte, lte = chipsim.subset(((xv @ W1.T + b1) >= 0).astype(np.int32),
                                      yv, cls)
            Wc, bc = chipsim.fit_head(ftr, ltr, epochs=args.epochs)
            a = ((fte @ Wc.T + bc).argmax(axis=1) == lte).mean()
            print(f"  A B C D E F (in EMNIST):     {a:.3f}")
            print(f"  {' '.join(names[:3])}... (not in EMNIST): {acc:.3f}")
        except Exception as e:
            print(f"  skipped: {e}")

    print("""
What this shows: the frozen half of the chip was trained only on handwritten
letters and digits, and cannot be changed. Ninety bytes of head weights are
enough to make it recognise shapes it never saw. If the accuracy above is
high, the features generalise beyond the training domain.
""")


if __name__ == "__main__":
    main()
