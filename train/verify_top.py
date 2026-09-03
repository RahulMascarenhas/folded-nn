#!/usr/bin/env python3
"""Verify the generated top-level RTL against numpy. Run before every push."""
import sys, json, subprocess, tempfile
from pathlib import Path
import numpy as np

def main(npz="weights_f56.npz", src="../src", n_img=8):
    meta = json.load(open("head_bitstream.json"))
    mod, nf, nc, bits = (meta["module"], meta["features"],
                         meta["classes"], meta["head_bits"])
    d = np.load(npz)
    W1 = d["W1"].astype(int)
    if W1.shape[1] != 64: W1 = W1.T
    b1 = d["b1"].astype(int)
    W2 = d["W2"].astype(int)
    if W2.shape[1] != nf: W2 = W2.T
    W2, b2 = W2[:nc], d["b2"].astype(int)[:nc]

    rng = np.random.default_rng(11)
    imgs = (rng.random((n_img, 64)) < 0.22).astype(int)
    gold = ((imgs @ W1.T + b1) >= 0).astype(int) @ W2.T + b2

    td = Path(tempfile.mkdtemp())
    for f in ("backbone.v", "head.v", f"{mod}.v"):
        (td / f).write_text((Path(src) / f).read_text())
    (td / "img.txt").write_text("\n".join(
        "".join(str(x) for x in reversed(im)) for im in imgs))
    (td / "hd.txt").write_text(meta["bitstream"] + "\n")
    (td / "tb.v").write_text(f'''`timescale 1ns/1ps
module tb;
 reg [7:0] ui=0; wire [7:0] uo,uio_out,uio_oe; reg clk=0,rst_n=0;
 {mod} dut(.ui_in(ui),.uo_out(uo),.uio_in(8'h0),.uio_out(uio_out),
   .uio_oe(uio_oe),.ena(1'b1),.clk(clk),.rst_n(rst_n));
 always #5 clk=~clk;
 reg [63:0] img [0:{n_img-1}]; reg [{bits-1}:0] hd [0:0]; integer t,i,g;
 initial begin
  $readmemb("img.txt",img); $readmemb("hd.txt",hd);
  @(negedge clk); rst_n=1; @(negedge clk);
  for(i={bits-1};i>=0;i=i-1) begin ui[3]=hd[0][i]; ui[4]=1'b1; @(negedge clk); end
  ui[4]=0; @(negedge clk);
  for(t=0;t<{n_img};t=t+1) begin
    for(i=63;i>=0;i=i-1) begin ui[0]=img[t][i]; ui[1]=1'b1; @(negedge clk); end
    ui[1]=0; @(negedge clk);
    ui[2]=1; @(negedge clk); ui[2]=0;
    g=0;
    while(!uio_out[1] && g<8000) begin
      @(posedge clk); #1;
      if (uio_out[0]) $display("S %0d %0d %0d", t, uio_out[5:3], $signed(uo));
      g=g+1;
    end
    @(negedge clk);
  end
  $finish; end
endmodule''')
    r = subprocess.run(f"cd {td} && iverilog -o s *.v && ./s",
                       shell=True, capture_output=True, text=True)
    got = np.zeros_like(gold); n = 0
    for line in r.stdout.split("\n"):
        if line.startswith("S "):
            _, t, c, v = line.split(); got[int(t), int(c)] = int(v); n += 1
    ok = n == gold.size and np.array_equal(got, gold)
    print(f"{mod}: {n}/{gold.size} scores, "
          f"{'BIT-EXACT' if ok else 'MISMATCH'}")
    if not ok:
        print("gold\n", gold, "\ngot\n", got)
        print(r.stderr[-800:])
    return ok

if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:]) else 1)
