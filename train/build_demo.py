#!/usr/bin/env python3
"""
Build a self-contained demo.html.

Embeds the frozen backbone weights and one or more head bitstreams into the
page, so it opens with a double-click and needs no server.

    python3 build_demo.py                       shipped head only
    python3 build_demo.py --heads heads/*.json  plus any you fitted
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BYCLASS = (
    [str(i) for i in range(10)]
    + [chr(ord("A") + i) for i in range(26)]
    + [chr(ord("a") + i) for i in range(26)]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE / "weights_f56.npz"))
    ap.add_argument("--template", default=str(HERE / "demo_template.html"))
    ap.add_argument("--heads", nargs="*", default=[])
    ap.add_argument("--out", default=str(HERE / "demo.html"))
    args = ap.parse_args()

    d = np.load(args.weights)
    W1, b1 = d["W1"].astype(int), d["b1"].astype(int)

    heads = {}
    meta_path = HERE / "head_bitstream.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        heads["shipped: 0 1 2 3 4 5"] = {
            "characters": BYCLASS[:6],
            "bitstream": m["bitstream"],
            "accuracy": 0.0,
        }

    paths = []
    for pat in args.heads:
        paths.extend(glob.glob(pat))
    for p in sorted(paths):
        h = json.loads(Path(p).read_text())
        heads[" ".join(h["characters"])] = {
            "characters": h["characters"],
            "bitstream": h["bitstream"],
            "accuracy": h.get("accuracy", 0.0),
        }

    if not heads:
        raise SystemExit("no heads found -- run chipsim.py --fit first")

    # The host-side classifier. Prefer a full-precision head if one has been
    # fitted -- software has no reason to be ternary, and the ternary version
    # is 63% zeros, which caps its scores and costs a lot of accuracy.
    soft = HERE / "software_head.json"
    if soft.exists():
        sh = json.loads(soft.read_text())
        W2 = np.array(sh["W"], dtype=float)
        b2 = np.array(sh["b"], dtype=float)
        soft_labels = sh["labels"]
        soft_acc = sh.get("accuracy", 0.0)
        soft_kind = f"full precision, {soft_acc:.3f}"
    else:
        W2 = d["W2"].astype(int)
        if W2.shape[1] != W1.shape[0]:
            W2 = W2.T
        b2 = d["b2"].astype(int)
        soft_labels = BYCLASS[: len(b2)]
        soft_kind = "ternary (run chipsim.py --fit-software for a better one)"

    payload = {
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "W2": W2.tolist(),
        "b2": b2.tolist(),
        "byclass": soft_labels,
        "heads": heads,
        "zero_rate": float((W1 == 0).mean()),
    }

    html = Path(args.template).read_text()
    inject = (
        "<script>window.CHIP_DATA = "
        + json.dumps(payload, separators=(",", ":"))
        + ";</script>"
    )
    if "<!--CHIP_DATA-->" not in html:
        raise SystemExit("template is missing the <!--CHIP_DATA--> marker")
    html = html.replace("<!--CHIP_DATA-->", inject, 1)

    Path(args.out).write_text(html)
    kb = len(html) / 1024
    print(f"wrote {args.out}  ({kb:.0f} KB, self-contained)")
    print(f"  backbone  {W1.shape[0]} features, {(W1 == 0).mean():.1%} zeros")
    print(f"  software  {len(b2)}-class head, {soft_kind}")
    print(f"  heads     {len(heads)}")
    for k, v in heads.items():
        acc = f"  {v['accuracy']:.3f}" if v["accuracy"] else ""
        print(f"            {k}{acc}")


if __name__ == "__main__":
    main()
