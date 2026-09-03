#!/usr/bin/env python3
"""Check the emitted serial backbone against a numpy golden model.
Run this before trusting any area number."""

import sys, subprocess, tempfile
from pathlib import Path
import numpy as np
import serial_sweep as S


def check(npz, lanes_list=None, n_img=8, seed=7):
    d = np.load(npz)
    W = d["W1"].astype(int)
    if W.shape[1] != 64:
        W = W.T
    b = d["b1"].astype(int) if "b1" in d else np.zeros(W.shape[0], int)
    nf = W.shape[0]
    rng = np.random.default_rng(seed)
    imgs = (rng.random((n_img, 64)) < 0.22).astype(int)
    gold = ((imgs @ W.T + b) >= 0).astype(int)
    lanes_list = lanes_list or [l for l in (7, 8, 14, 28, nf) if nf % l == 0]
    ok = True
    for lanes in lanes_list:
        v, passes = S.emit_serial_backbone(W, b, lanes)
        td = Path(tempfile.mkdtemp())
        (td / "bb.v").write_text(v)
        (td / "tb.v").write_text(f"""`timescale 1ns/1ps
module tb;
  reg clk=0, rst_n=0, start=0, pix=0;
  wire [{nf - 1}:0] features; wire done;
  backbone u(.clk(clk),.rst_n(rst_n),.start(start),.pix(pix),.features(features),.done(done));
  always #5 clk = ~clk;
  reg [63:0] img [0:{n_img - 1}];
  integer t,p,i;
  initial begin
    $readmemb("img.txt", img);
    @(negedge clk); rst_n = 1;
    for (t=0;t<{n_img};t=t+1) begin
      @(negedge clk); start=1; @(negedge clk); start=0;
      for (p=0;p<{passes};p=p+1)
        for (i=0;i<64;i=i+1) begin pix=img[t][i]; @(negedge clk); end
      $display("R %b", features);
    end
    $finish;
  end
endmodule""")
        (td / "img.txt").write_text(
            "\n".join("".join(str(x) for x in reversed(im)) for im in imgs)
        )
        r = subprocess.run(
            f"cd {td} && iverilog -o s bb.v tb.v && ./s",
            shell=True,
            capture_output=True,
            text=True,
        )
        out = [l.split()[1] for l in r.stdout.split("\n") if l.startswith("R ")]
        if len(out) != n_img:
            print(f"  lanes={lanes:>3}  SIM FAILED")
            print(r.stderr[-300:])
            ok = False
            continue
        got = np.array([[int(c) for c in reversed(s)] for s in out])
        match = np.array_equal(got, gold)
        ok &= match
        print(
            f"  lanes={lanes:>3}  passes={passes}  "
            f"{'OK' if match else 'MISMATCH'}  "
            f"errors {(got != gold).sum()}/{gold.size}"
        )
    return ok


if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else "weights_f56.npz"
    print(f"verifying {npz} against numpy golden model")
    sys.exit(0 if check(npz) else 1)
