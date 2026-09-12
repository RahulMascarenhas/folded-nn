#!/usr/bin/env python3
"""
Bit-exact software model of the chip, plus a head fitter.

Two jobs:

  1. mock the silicon exactly -- same integer arithmetic, same cycle counts,
     same bitstream format -- so you can test without hardware
  2. fit a new 6-class head onto the frozen backbone, including on classes
     the backbone never saw

    python3 chipsim.py --list                       what the shipped head does
    python3 chipsim.py --fit 0 1 2 3 4 5            digits
    python3 chipsim.py --fit A B C D E F            letters
    python3 chipsim.py --fit @ '#' '$' '%' '&' '?'  anything in EMNIST byclass
    python3 chipsim.py --fit-random 5               five random class sets

Every fitted head prints its accuracy and writes a 90-byte bitstream you can
shift into the real chip.
"""

import argparse
import json
from pathlib import Path

import numpy as np

PIXELS, FEATURES, CLASSES = 64, 56, 6
LANES, PASSES = 8, 7
ACC_W, BIAS_W = 9, 8
HEAD_BITS = 2 * FEATURES * CLASSES + BIAS_W * CLASSES  # 720

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights_f56.npz"

# EMNIST byclass label order: 0-9, A-Z, a-z
BYCLASS = (
    [str(i) for i in range(10)]
    + [chr(ord("A") + i) for i in range(26)]
    + [chr(ord("a") + i) for i in range(26)]
)


# --------------------------------------------------------- preprocessing


def centre(imgs):
    """Shift each 8x8 image so its centre of mass sits at the middle.

    EMNIST is size-normalised and centred by NIST, so the frozen features
    never had to learn translation invariance -- and they did not. Anything
    fed to this chip must be centred the same way or accuracy collapses.
    Measured: +12 to +19 points on out-of-domain symbols.
    """
    a = np.atleast_2d(np.asarray(imgs, dtype=np.int32))
    out = np.zeros_like(a)
    for i, v in enumerate(a):
        g = v.reshape(8, 8)
        if g.sum() == 0:
            out[i] = v
            continue
        ys, xs = np.nonzero(g)
        dy = int(round(3.5 - ys.mean()))
        dx = int(round(3.5 - xs.mean()))
        h = np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        if dy > 0:
            h[:dy] = 0
        elif dy < 0:
            h[dy:] = 0
        if dx > 0:
            h[:, :dx] = 0
        elif dx < 0:
            h[:, dx:] = 0
        out[i] = h.reshape(64)
    return out[0] if np.ndim(imgs) == 1 else out


# ------------------------------------------------------------- the chip


class Chip:
    """Integer model of the silicon. No floats anywhere in the datapath."""

    def __init__(self, W1, b1):
        self.W1 = np.asarray(W1, dtype=np.int32)  # (56, 64) ternary
        self.b1 = np.asarray(b1, dtype=np.int32)
        self.head = None  # loaded over SPI

    # --- the frozen half, cast into logic -----------------------------

    def features(self, img64):
        """8 accumulators over 7 passes, 448 cycles. Sign bit is the output."""
        img = np.asarray(img64, dtype=np.int32).reshape(PIXELS)
        out = np.zeros(FEATURES, dtype=np.int32)
        for p in range(PASSES):
            for l in range(LANES):
                f = p * LANES + l
                acc = int(self.b1[f])
                for i in range(PIXELS):
                    if img[i]:
                        acc += int(self.W1[f, i])
                self._check(acc, ACC_W, f"backbone acc f{f}")
                out[f] = 1 if acc >= 0 else 0
        return out

    # --- the loadable half --------------------------------------------

    def load_head(self, bitstream):
        """Shift 720 bits in, MSB first, exactly as the hardware does."""
        if len(bitstream) != HEAD_BITS:
            raise ValueError(f"expected {HEAD_BITS} bits, got {len(bitstream)}")
        reg = int(bitstream, 2)
        bits = [(reg >> k) & 1 for k in range(HEAD_BITS)]

        W2 = np.zeros((CLASSES, FEATURES), dtype=np.int32)
        for c in range(CLASSES):
            for f in range(FEATURES):
                off = 2 * (c * FEATURES + f)
                code = bits[off] | (bits[off + 1] << 1)
                W2[c, f] = {0: 0, 1: 1, 2: -1, 3: 0}[code]

        b2 = np.zeros(CLASSES, dtype=np.int32)
        for c in range(CLASSES):
            off = 2 * FEATURES * CLASSES + BIAS_W * c
            v = sum(bits[off + k] << k for k in range(BIAS_W))
            b2[c] = v - 256 if v > 127 else v
        self.head = (W2, b2)

    def scores(self, img64):
        """Six signed 8-bit scores, in the order the chip emits them."""
        if self.head is None:
            raise RuntimeError("no head loaded")
        W2, b2 = self.head
        feats = self.features(img64)
        out = np.zeros(CLASSES, dtype=np.int32)
        for c in range(CLASSES):
            acc = int(b2[c])
            for f in range(FEATURES):
                if feats[f]:
                    acc += int(W2[c, f])
            self._check(acc, BIAS_W, f"head acc c{c}")
            out[c] = acc
        return out

    def classify(self, img64):
        """argmax, which the real chip leaves to the host."""
        return int(np.argmax(self.scores(img64)))

    @staticmethod
    def _check(v, width, where):
        lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
        if not lo <= v <= hi:
            raise OverflowError(
                f"{where}: {v} outside {width}-bit range "
                f"[{lo}, {hi}] -- would wrap in silicon"
            )

    cycles = PIXELS * PASSES + FEATURES * CLASSES  # 784


# ------------------------------------------------------------ bitstream


def pack_head(W2, b2):
    """Weights and biases -> 720 bits, MSB first."""
    bits = [0] * HEAD_BITS
    for c in range(CLASSES):
        for f in range(FEATURES):
            code = {0: 0b00, 1: 0b01, -1: 0b10}[int(W2[c, f])]
            off = 2 * (c * FEATURES + f)
            bits[off] = code & 1
            bits[off + 1] = (code >> 1) & 1
    for c in range(CLASSES):
        v = int(b2[c]) & 0xFF
        off = 2 * FEATURES * CLASSES + BIAS_W * c
        for k in range(BIAS_W):
            bits[off + k] = (v >> k) & 1
    return "".join(str(b) for b in reversed(bits))


# ------------------------------------------------------- fitting a head


def ternarise(w, scale=0.7):
    delta = scale * np.mean(np.abs(w))
    return np.where(np.abs(w) > delta, np.sign(w), 0.0).astype(np.float32)


def fit_head(feats, labels, epochs=400, lr=0.02, thresh=0.5, seed=0):
    """Train 336 ternary weights + 6 biases on frozen features.

    Straight-through estimator, same as the backbone was trained with. The
    biases are clipped to 8 bits because that is what the hardware holds.

    Note on `thresh`: in the backbone a zero weight disappears at synthesis
    and saves silicon. In the head it does not -- every weight is 2 bits of
    flip-flop regardless of value -- so there is no reason to sparsify here.
    """
    rng = np.random.default_rng(seed)
    n_cls = int(labels.max()) + 1
    W = rng.normal(0, 0.5, (FEATURES, n_cls)).astype(np.float32)
    b = np.zeros(n_cls, dtype=np.float32)
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    mb = np.zeros_like(b)
    vb = np.zeros_like(b)

    x = feats.astype(np.float32)
    best, best_wb = -1.0, (W.copy(), b.copy())

    for ep in range(1, epochs + 1):
        Wq = ternarise(W, thresh)
        logits = x @ Wq + b
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        d = p.copy()
        d[np.arange(len(labels)), labels] -= 1.0
        d /= len(labels)

        gW, gb = x.T @ d, d.sum(axis=0)
        for arr, g, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            arr -= lr * (m / (1 - 0.9**ep)) / (np.sqrt(v / (1 - 0.999**ep)) + 1e-8)
        np.clip(W, -1.5, 1.5, out=W)
        np.clip(b, -120, 120, out=b)

        if ep % 20 == 0 or ep == epochs:
            acc = ((x @ ternarise(W, thresh) + b).argmax(axis=1) == labels).mean()
            if acc > best:
                best, best_wb = acc, (W.copy(), b.copy())

    W, b = best_wb
    return ternarise(W, thresh).T.astype(np.int32), np.round(b).astype(np.int32)


# ------------------------------------------------------------------ data


def load_emnist(split="byclass", pix=0.25, chunk=20000):
    """Load from ~/.cache/emnist-idx/.

    Chunked: byclass has ~700k images, and converting them all to float32 at
    once needs several GB.
    """
    import gzip

    cache = Path.home() / ".cache" / "emnist-idx"

    def read(name):
        with gzip.open(cache / name, "rb") as f:
            magic = f.read(4)
            dims = [int.from_bytes(f.read(4), "big") for _ in range(magic[3])]
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)

    def to8(a):
        n = len(a)
        out = np.empty((n, 64), dtype=np.int32)
        for i in range(0, n, chunk):
            c = a[i : i + chunk].transpose(0, 2, 1).astype(np.float32) / 255.0
            m = len(c)
            c = np.pad(c, ((0, 0), (2, 2), (2, 2)))
            c = c.reshape(m, 8, 4, 8, 4).mean(axis=(2, 4))
            out[i : i + m] = (c > pix).astype(np.int32).reshape(m, 64)
        return out

    return (
        to8(read(f"emnist-{split}-train-images-idx3-ubyte.gz")),
        read(f"emnist-{split}-train-labels-idx1-ubyte.gz").astype(int),
        to8(read(f"emnist-{split}-test-images-idx3-ubyte.gz")),
        read(f"emnist-{split}-test-labels-idx1-ubyte.gz").astype(int),
    )


def subset(x, y, classes):
    mask = np.isin(y, classes)
    remap = {c: i for i, c in enumerate(classes)}
    return x[mask], np.array([remap[v] for v in y[mask]])


def fit_software_head(feats, labels, n_cls, epochs=300, lr=0.05, seed=0):
    """Full-precision head for the host.

    The on-chip head must be ternary because every weight is 2 bits of
    flip-flop. A head that runs in software has no such constraint, so
    ternarising it throws away capacity for nothing.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, (FEATURES, n_cls)).astype(np.float32)
    b = np.zeros(n_cls, dtype=np.float32)
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    mb = np.zeros_like(b)
    vb = np.zeros_like(b)
    x = feats.astype(np.float32)
    best, snap = -1.0, (W.copy(), b.copy())

    for ep in range(1, epochs + 1):
        logits = x @ W + b
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        d = p.copy()
        d[np.arange(len(labels)), labels] -= 1.0
        d /= len(labels)
        gW, gb = x.T @ d, d.sum(axis=0)
        for arr, g, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            arr -= lr * (m / (1 - 0.9**ep)) / (np.sqrt(v / (1 - 0.999**ep)) + 1e-8)
        if ep % 20 == 0 or ep == epochs:
            acc = ((x @ W + b).argmax(axis=1) == labels).mean()
            if acc > best:
                best, snap = acc, (W.copy(), b.copy())
    W, b = snap
    return W.T, b, best


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fit", nargs=6, metavar="CHAR", help="six characters to retarget the head to"
    )
    ap.add_argument(
        "--fit-random", type=int, metavar="N", help="fit N random 6-class sets"
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="show the shipped head and available characters",
    )
    ap.add_argument(
        "--fit-software",
        action="store_true",
        help="fit a full-precision 62-class head for the host",
    )
    ap.add_argument("--out", default="heads")
    ap.add_argument("--epochs", type=int, default=400)
    args = ap.parse_args()

    d = np.load(WEIGHTS)
    chip = Chip(d["W1"], d["b1"])

    if args.list:
        meta = json.loads((HERE / "head_bitstream.json").read_text())
        print(f"module       {meta['module']}")
        print(f"features     {FEATURES}   ({LANES} lanes, {PASSES} passes)")
        print(f"head         {HEAD_BITS} bits = {HEAD_BITS // 8} bytes")
        print(f"cycles       {Chip.cycles} per inference")
        print(f"shipped head classes: {' '.join(BYCLASS[i] for i in range(CLASSES))}")
        print(f"\navailable characters ({len(BYCLASS)}):")
        print("  " + " ".join(BYCLASS))
        return

    if args.fit_software:
        print("loading emnist byclass ...")
        xtr, ytr, xte, yte = load_emnist()
        n_cls = int(max(ytr.max(), yte.max())) + 1
        print("computing frozen features ...")
        F = (xtr @ chip.W1.T + chip.b1 >= 0).astype(np.float32)
        Fte = (xte @ chip.W1.T + chip.b1 >= 0).astype(np.float32)
        print(f"fitting {n_cls}-class head, full precision ...")
        W, b, tr_acc = fit_software_head(F, ytr, n_cls)
        acc = ((Fte @ W.T + b).argmax(axis=1) == yte).mean()
        out = HERE / "software_head.json"
        out.write_text(
            json.dumps(
                {
                    "classes": n_cls,
                    "accuracy": float(acc),
                    "labels": BYCLASS[:n_cls],
                    "W": [[round(float(v), 4) for v in row] for row in W],
                    "b": [round(float(v), 4) for v in b],
                }
            )
        )
        print(f"\n  test accuracy {acc:.4f}  (ternary version was 0.449)")
        print(f"  wrote {out}")
        print("\n  build_demo.py will embed this instead of the ternary head")
        return

    targets = []
    if args.fit:
        targets.append(args.fit)
    if args.fit_random:
        rng = np.random.default_rng(0)
        for k in range(args.fit_random):
            idx = rng.choice(len(BYCLASS), CLASSES, replace=False)
            targets.append([BYCLASS[i] for i in sorted(idx)])
    if not targets:
        ap.error("give --fit, --fit-random or --list")

    print("loading emnist byclass ...")
    xtr, ytr, xte, yte = load_emnist()

    print("computing frozen features (the part cast into silicon) ...")
    F = (xtr @ chip.W1.T + chip.b1 >= 0).astype(np.int32)
    Fte = (xte @ chip.W1.T + chip.b1 >= 0).astype(np.int32)

    Path(args.out).mkdir(exist_ok=True)
    print(f"\n{'characters':>20} {'accuracy':>9} {'bitstream':>28}")
    print("-" * 62)

    for chars in targets:
        try:
            classes = [BYCLASS.index(c) for c in chars]
        except ValueError as e:
            print(f"  unknown character: {e}")
            continue

        ftr, ltr = subset(F, ytr, classes)
        fte, lte = subset(Fte, yte, classes)
        W2, b2 = fit_head(ftr, ltr, epochs=args.epochs)

        pred = (fte @ W2.T + b2).argmax(axis=1)
        acc = (pred == lte).mean()

        stream = pack_head(W2, b2)
        name = "".join(c if c.isalnum() else f"u{ord(c)}" for c in chars)
        path = Path(args.out) / f"head_{name}.json"
        path.write_text(
            json.dumps(
                {
                    "characters": chars,
                    "classes": classes,
                    "accuracy": float(acc),
                    "bits": HEAD_BITS,
                    "bytes": HEAD_BITS // 8,
                    "bitstream": stream,
                    "hex": f"{int(stream, 2):0{HEAD_BITS // 4}x}",
                },
                indent=2,
            )
        )

        # confirm the packed bitstream round-trips through the chip model
        chip.load_head(stream)
        n = min(200, len(fte))
        sim = np.array([chip.classify(x) for x in subset(xte, yte, classes)[0][:n]])
        agree = (sim == pred[:n]).mean()
        flag = "" if agree == 1.0 else f"  MISMATCH {agree:.1%}"

        print(f"{' '.join(chars):>20} {acc:>9.3f} {path.name:>28}{flag}")

    print(f"\nEach file holds a {HEAD_BITS // 8}-byte bitstream. Shift it in MSB")
    print("first on ui_in[3] with ui_in[4] high, and the chip recognises those")
    print("six characters instead. The frozen half never changes.")


if __name__ == "__main__":
    main()
