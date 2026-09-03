#!/usr/bin/env python3
"""
Ternary quantisation-aware trainer for a frozen-backbone classifier.

    64 binary pixels
      -> backbone   64 -> FEATURES   ternary weights, sign activation
      -> head       FEATURES -> CLASSES   ternary weights + integer bias

Trains with a straight-through estimator so the network learns *with* the
ternary weights and the sign step in place, rather than being rounded off
afterwards.

Outputs weights_f{N}.npz per feature width, which feeds directly into the
area sweep. Also prints the zero rate, which is the number that actually
determines silicon area.

Quick smoke test (no download, runs in seconds):
    python3 train.py --dataset digits --features 32 --epochs 10

Real run:
    pip install emnist
    python3 train.py --dataset emnist --features 32 40 48 56

Sparsity is a design knob -- more zeros means less silicon:
    python3 train.py --dataset emnist --features 48 --thresh 0.9 --l1 1e-4
"""

import argparse
import gzip
import sys
import urllib.request
from pathlib import Path

import numpy as np


# --------------------------------------------------------------- data

def to_8x8_binary(images, threshold=0.15, chunk=20000):
    """28x28 greyscale -> 8x8 binary.

    Pad to 32x32 then average-pool 4x4, which divides evenly and keeps the
    whole glyph. Then threshold at a fraction of full scale.

    Chunked, because byclass has 814k images and converting them all to
    float32 at once needs several GB.
    """
    n = images.shape[0]
    out = np.empty((n, 64), dtype=np.float32)
    for i in range(0, n, chunk):
        x = images[i:i + chunk].astype(np.float32) / 255.0
        m = x.shape[0]
        x = np.pad(x, ((0, 0), (2, 2), (2, 2)))          # 28 -> 32
        x = x.reshape(m, 8, 4, 8, 4).mean(axis=(2, 4))   # 32 -> 8
        out[i:i + m] = (x > threshold).astype(np.float32).reshape(m, 64)
    return out


def show(img64, label=""):
    """ASCII art of one 8x8 image, to eyeball orientation."""
    g = img64.reshape(8, 8)
    print(f"  sample: {label}")
    for row in g:
        print("    " + "".join("##" if v else ". " for v in row))


HF_BASE = "https://huggingface.co/datasets/Heliosoph/EMNIST/resolve/main"
CACHE = Path.home() / ".cache" / "emnist-idx"


def read_idx(path):
    """Parse an IDX file (the MNIST binary format). Handles .gz transparently."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        magic = f.read(4)
        if magic[0] or magic[1]:
            raise ValueError(f"{path}: not an IDX file")
        ndim = magic[3]
        dims = [int.from_bytes(f.read(4), "big") for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(dims)


def fetch_idx(name):
    """Download one IDX file from the mirror, cached locally."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / name
    if dest.exists() and dest.stat().st_size > 1000:
        return dest

    url = f"{HF_BASE}/{name}"
    print(f"  fetching {name} ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        dest.unlink(missing_ok=True)
        sys.exit(f"download failed: {e}\n  try manually:  curl -L -o {dest} {url}")

    if dest.stat().st_size < 1000:   # an error page, not data
        dest.unlink()
        sys.exit(f"got a tiny file from {url} -- the mirror may have moved")
    return dest


def load_emnist_hf(split="balanced", transpose=True, pix=0.15):
    """Load EMNIST from local cache or the HuggingFace mirror. Converts each
    array to 8x8 immediately so the large uint8 arrays do not pile up."""
    def get(kind, usage):
        a = read_idx(fetch_idx(f"emnist-{split}-{usage}-{kind}"))
        if kind.startswith("idx3") and transpose:
            a = a.transpose(0, 2, 1)
        return a

    xtr = to_8x8_binary(get("idx3-ubyte.gz", "train-images"), pix)
    ytr = read_idx(fetch_idx(f"emnist-{split}-train-labels-idx1-ubyte.gz"))
    xte = to_8x8_binary(get("idx3-ubyte.gz", "test-images"), pix)
    yte = read_idx(fetch_idx(f"emnist-{split}-test-labels-idx1-ubyte.gz"))

    return xtr, ytr.astype(int), xte, yte.astype(int)


def load_emnist(split="balanced", transpose=True):
    try:
        from emnist import extract_training_samples, extract_test_samples
    except ImportError:
        sys.exit("pip install emnist   (see the notes at the bottom of this file)")

    xtr, ytr = extract_training_samples(split)
    xte, yte = extract_test_samples(split)

    # EMNIST ships rotated relative to MNIST. If the ASCII art below looks
    # sideways or mirrored, flip this with --no-transpose.
    if transpose:
        xtr = xtr.transpose(0, 2, 1)
        xte = xte.transpose(0, 2, 1)

    return to_8x8_binary(xtr), ytr, to_8x8_binary(xte), yte


def load_digits_8x8():
    """sklearn's 8x8 digits -- already the right size. For smoke testing."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits()
    x = (d.images / 16.0 > 0.3).astype(np.float32).reshape(-1, 64)
    return train_test_split(x, d.target, test_size=0.25, random_state=0)


# ------------------------------------------------------- quantisation

def ternarise(w, thresh_scale=0.7):
    """Weights below a threshold become 0; the rest become +1 or -1.

    Raising thresh_scale produces more zeros, which is less silicon and
    slightly less accuracy.
    """
    delta = thresh_scale * np.mean(np.abs(w))
    return np.where(np.abs(w) > delta, np.sign(w), 0.0).astype(np.float32)


class Adam:
    """Plain Adam. Note the two moment arrays are allocated separately --
    writing `m = v = np.zeros_like(w)` aliases them and silently drives every
    weight to zero. The symptom is accuracy pinned at exactly chance."""

    def __init__(self, shape, lr=3e-2):
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.lr, self.t = lr, 0

    def step(self, w, g, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        self.m = b1 * self.m + (1 - b1) * g
        self.v = b2 * self.v + (1 - b2) * g * g
        mh = self.m / (1 - b1 ** self.t)
        vh = self.v / (1 - b2 ** self.t)
        return w - self.lr * mh / (np.sqrt(vh) + eps)


# ---------------------------------------------------------- the model

class Net:
    def __init__(self, n_feat, n_class, thresh=0.7, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (64, n_feat)).astype(np.float32)
        self.b1 = np.zeros(n_feat, dtype=np.float32)
        self.W2 = rng.normal(0, 0.5, (n_feat, n_class)).astype(np.float32)
        self.b2 = np.zeros(n_class, dtype=np.float32)
        self.thresh = thresh
        self.opt = {k: Adam(getattr(self, k).shape)
                    for k in ("W1", "b1", "W2", "b2")}

    def forward(self, x, train=True):
        W1q = ternarise(self.W1, self.thresh)
        W2q = ternarise(self.W2, self.thresh)

        a1 = x @ W1q + self.b1
        # sign activation: +1 / -1 in the forward pass
        h = np.where(a1 >= 0, 1.0, -1.0).astype(np.float32)
        logits = h @ W2q + self.b2

        if train:
            self.cache = (x, a1, h, W1q, W2q)
        return logits

    def backward(self, logits, y, l1=0.0):
        x, a1, h, W1q, W2q = self.cache
        n = x.shape[0]

        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        d = p.copy()
        d[np.arange(n), y] -= 1.0
        d /= n

        gW2 = h.T @ d
        gb2 = d.sum(axis=0)

        dh = d @ W2q.T
        # straight-through: pass the gradient where the pre-activation was
        # inside [-1, 1], block it outside
        da1 = dh * (np.abs(a1) <= 1.0)

        gW1 = x.T @ da1
        gb1 = da1.sum(axis=0)

        if l1:  # pushes weights toward zero -> more zeros -> less silicon
            gW1 += l1 * np.sign(self.W1)
            gW2 += l1 * np.sign(self.W2)

        self.W1 = self.opt["W1"].step(self.W1, gW1)
        self.b1 = self.opt["b1"].step(self.b1, gb1)
        self.W2 = self.opt["W2"].step(self.W2, gW2)
        self.b2 = self.opt["b2"].step(self.b2, gb2)

        # latent weights must stay bounded or the threshold drifts
        np.clip(self.W1, -1.5, 1.5, out=self.W1)
        np.clip(self.W2, -1.5, 1.5, out=self.W2)

    def snapshot(self):
        return (self.W1.copy(), self.b1.copy(),
                self.W2.copy(), self.b2.copy())

    def restore(self, snap):
        self.W1, self.b1, self.W2, self.b2 = [a.copy() for a in snap]

    def accuracy(self, x, y, batch=4096):
        correct = 0
        for i in range(0, len(x), batch):
            lo = self.forward(x[i:i + batch], train=False)
            correct += (lo.argmax(axis=1) == y[i:i + batch]).sum()
        return correct / len(x)

    def quantised(self):
        return (ternarise(self.W1, self.thresh), self.b1,
                ternarise(self.W2, self.thresh), self.b2)


def train(net, xtr, ytr, xte, yte, epochs, batch, l1, seed=0):
    """Keeps the best epoch. Accuracy oscillates, so the last one is often
    not the one you want in silicon."""
    rng = np.random.default_rng(seed)
    n = len(xtr)
    best_acc, best_snap = 0.0, net.snapshot()
    for ep in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            net.backward(net.forward(xtr[idx]), ytr[idx], l1=l1)
        if ep % max(1, epochs // 20) == 0 or ep == epochs - 1:
            acc = net.accuracy(xte, yte)
            if acc > best_acc:
                best_acc, best_snap = acc, net.snapshot()
            if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
                print(f"    epoch {ep:>3}  test {acc:.4f}"
                      f"{'  *' if acc == best_acc else ''}")
    net.restore(best_snap)
    return best_acc


# -------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["emnist", "digits"], default="emnist")
    ap.add_argument("--source", choices=["hf", "pkg"], default="hf",
                    help="hf = HuggingFace mirror (works); pkg = emnist package (NIST url is dead)")
    ap.add_argument("--split", default="balanced",
                    help="emnist split: balanced (47 classes), byclass, digits")
    ap.add_argument("--features", type=int, nargs="+", default=[32, 40, 48, 56])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--thresh", type=float, default=0.7,
                    help="ternary threshold scale; higher = more zeros")
    ap.add_argument("--l1", type=float, default=0.0,
                    help="L1 penalty; higher = more zeros")
    ap.add_argument("--no-transpose", action="store_true")
    ap.add_argument("--pix-thresh", type=float, default=0.15,
                    help="pixel binarisation threshold; higher = thinner strokes")
    ap.add_argument("--out-prefix", default="weights")
    args = ap.parse_args()

    if args.dataset == "emnist":
        print(f"loading emnist '{args.split}' via {args.source} ...")
        if args.source == "hf":
            xtr, ytr, xte, yte = load_emnist_hf(
                args.split, not args.no_transpose, args.pix_thresh)
        else:
            xtr, ytr, xte, yte = load_emnist(args.split, not args.no_transpose)
    else:
        print("loading sklearn digits (smoke test) ...")
        xtr, xte, ytr, yte = load_digits_8x8()

    n_class = int(max(ytr.max(), yte.max())) + 1
    print(f"  train {len(xtr)}  test {len(xte)}  classes {n_class}")
    for lab in range(min(3, n_class)):
        i = int(np.where(ytr == lab)[0][0])
        show(xtr[i], f"class {lab}   (ink {xtr[i].mean():.0%})")
    print("  Classes 0-9 are digits 0-9. If they look sideways or mirrored,")
    print("  rerun with --no-transpose. If they look like solid blobs,")
    print("  raise --pix-thresh (try 0.3, 0.4, 0.5).\n")

    print(f"{'feat':>5} {'accuracy':>9} {'zeros W1':>9} {'zeros W2':>9} {'file':>22}")
    print("-" * 60)

    for f in args.features:
        print(f"  training {f} features ...")
        net = Net(f, n_class, thresh=args.thresh)
        for k in net.opt:
            net.opt[k].lr = args.lr
        acc = train(net, xtr, ytr, xte, yte, args.epochs, args.batch, args.l1)

        W1q, b1, W2q, b2 = net.quantised()
        z1 = float((W1q == 0).mean())
        z2 = float((W2q == 0).mean())

        path = f"{args.out_prefix}_f{f}.npz"
        np.savez(path, W1=W1q.T.astype(np.int8), b1=np.round(b1).astype(np.int32),
                 W2=W2q.T.astype(np.int8), b2=np.round(b2).astype(np.int32),
                 features=f, classes=n_class, accuracy=acc,
                 zero_rate_backbone=z1)

        print(f"{f:>5} {acc:>9.4f} {z1:>8.1%} {z2:>9.1%} {path:>22}\n")

    print("""
The zero rate is the number that drives silicon area. Every zero deletes an
input and shrinks the adder behind it.

To trade accuracy for area, raise --thresh (0.8, 0.9, 1.0) or add --l1 1e-4.
Watch what each point of accuracy buys you in the area sweep.

Sanity check: accuracy pinned at exactly 1/n_classes means a plumbing bug,
not a hard problem.
""")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
# Where the data comes from
#
#   pip install emnist
#
# First call downloads ~500 MB and caches it in ~/.cache/emnist/.
# If that mirror is down, fetch the archive directly:
#
#   https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip
#
# and unzip it to ~/.cache/emnist/, or grab the Kaggle mirror
# (crawford/emnist). The 'balanced' split is the one you want: 47 classes,
# 112,800 training images, digits and letters with confusable pairs merged.
# ---------------------------------------------------------------------
