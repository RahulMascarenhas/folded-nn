#!/usr/bin/env python3
"""
Area sweep for the serial backbone, using real trained weights.

Emits a bit-serial backbone where one pixel arrives per cycle and LANES
accumulators are reused across several passes. The weights stay constants
baked into the netlist; what changes with LANES is how much arithmetic is
built versus how much selection logic is needed.

    LANES = FEATURES   one pass, one accumulator per feature (the parallel
                       design, largest)
    LANES = 8          eight passes, smallest arithmetic, most selection

Somewhere between those the total is minimised. This finds it.

    python3 serial_sweep.py --npz weights_f56.npz
    python3 serial_sweep.py --npz weights_f56.npz --lanes 8 14 28 56
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PIXELS = 64
ACC_W = 9
NAND2_UM2 = 7.2576  # sg13g2_nand2_1

LIB_URL = (
    "https://raw.githubusercontent.com/IHP-GmbH/IHP-Open-PDK/main/"
    "ihp-sg13g2/libs.ref/sg13g2_stdcell/lib/"
    "sg13g2_stdcell_typ_1p20V_25C.lib"
)
LIB_CACHE = Path("sg13g2_stdcell_typ_1p20V_25C.lib")


# --------------------------------------------------------------- emit


def emit_serial_backbone(W, biases, lanes):
    """W: (FEATURES, 64) of -1/0/+1.

    One pixel per cycle into `lanes` accumulators, repeated over passes.
    Each lane's weights become two constant bit vectors indexed by a
    counter -- the synthesiser folds and minimises them.
    """
    n_feat, _ = W.shape
    if n_feat % lanes:
        raise ValueError(f"{n_feat} features not divisible by {lanes} lanes")
    passes = n_feat // lanes
    pass_w = max(1, (passes - 1).bit_length())
    rom_len = PIXELS * (1 << pass_w)  # padded; unused entries fold away

    L = []
    L.append(f"// {n_feat} features, {lanes} lanes, {passes} passes")
    L.append(f"module backbone (")
    L.append(f"    input  wire clk,")
    L.append(f"    input  wire rst_n,")
    L.append(f"    input  wire start,")
    L.append(f"    input  wire pix,")
    L.append(f"    output reg  [{n_feat - 1}:0] features,")
    L.append(f"    output reg  done")
    L.append(f");")
    L.append(f"    reg [5:0] idx;")
    L.append(f"    reg [{pass_w - 1}:0] pnum;")
    L.append(f"    wire [{pass_w + 5}:0] rom = {{pnum, idx}};")
    L.append(f"    reg busy;")
    L.append(f"    wire [{pass_w - 1}:0] pnum_next = pnum + {pass_w}'d1;")

    # weight ROMs: one +1 vector and one -1 vector per lane
    for l in range(lanes):
        pos = ["0"] * rom_len
        neg = ["0"] * rom_len
        for p in range(passes):
            f = p * lanes + l
            for i in range(PIXELS):
                w = W[f, i]
                if w == 1:
                    pos[p * PIXELS + i] = "1"
                elif w == -1:
                    neg[p * PIXELS + i] = "1"
        L.append(
            f"    localparam [{rom_len - 1}:0] POS{l} = "
            f"{rom_len}'b{''.join(reversed(pos))};"
        )
        L.append(
            f"    localparam [{rom_len - 1}:0] NEG{l} = "
            f"{rom_len}'b{''.join(reversed(neg))};"
        )

    for l in range(lanes):
        L.append(f"    reg signed [{ACC_W - 1}:0] acc{l};")

    # next-value wires: the last pixel of a pass must be included in the
    # feature that gets latched on that same cycle
    for l in range(lanes):
        L.append(
            f"    wire signed [{ACC_W - 1}:0] nxt{l} = acc{l}"
            f" + ((pix & POS{l}[rom]) ? {ACC_W}'sd1 :"
            f" (pix & NEG{l}[rom]) ? -{ACC_W}'sd1 : {ACC_W}'sd0);"
        )

    def bias_lit(f):
        b = int(biases[f])
        return f"-$signed({ACC_W}'sd{-b})" if b < 0 else f"$signed({ACC_W}'sd{b})"

    def bias_mux(l):
        """Bias for lane l depends on which pass we are about to start."""
        if passes == 1:
            return bias_lit(l)
        parts = [
            f"({pass_w}'d{p} == pnum_next) ? {bias_lit(p * lanes + l)} : "
            for p in range(passes - 1)
        ]
        return "".join(parts) + bias_lit((passes - 1) * lanes + l)

    L.append("    always @(posedge clk) begin")
    L.append("        if (!rst_n) begin")
    L.append("            idx <= 0; pnum <= 0; busy <= 0; done <= 0;")
    L.append("            features <= 0;")
    for l in range(lanes):
        L.append(f"            acc{l} <= 0;")
    L.append("        end else if (start) begin")
    L.append("            idx <= 0; pnum <= 0; busy <= 1; done <= 0;")
    for l in range(lanes):
        L.append(f"            acc{l} <= {bias_lit(l)};")
    L.append("        end else if (busy) begin")
    L.append("            if (idx == 6'd63) begin")
    L.append("                case (pnum)")
    for p in range(passes):
        L.append(f"                    {pass_w}'d{p}: begin")
        for l in range(lanes):
            L.append(
                f"                        features[{p * lanes + l}]"
                f" <= ~nxt{l}[{ACC_W - 1}];"
            )
        L.append("                    end")
    L.append("                    default: ;")
    L.append("                endcase")
    for l in range(lanes):
        L.append(f"                acc{l} <= {bias_mux(l)};")
    L.append(f"                if (pnum == {pass_w}'d{passes - 1}) begin")
    L.append("                    busy <= 0; done <= 1;")
    L.append("                end else pnum <= pnum + 1;")
    L.append("                idx <= 0;")
    L.append("            end else begin")
    for l in range(lanes):
        L.append(f"                acc{l} <= nxt{l};")
    L.append("                idx <= idx + 1;")
    L.append("            end")
    L.append("        end else done <= 0;")
    L.append("    end")
    L.append("endmodule")
    return "\n".join(L) + "\n", passes


def emit_head(n_feat, n_class, bias_w=8):
    n_w = n_feat * n_class
    bits = 2 * n_w + bias_w * n_class
    iw = max(1, (max(n_w, 2) - 1).bit_length())
    cw = max(1, (max(n_class, 2) - 1).bit_length())
    return f"""module head (
    input  wire clk, rst_n, load_en, load_bit, start,
    input  wire [{n_feat - 1}:0] features,
    output reg  signed [{bias_w - 1}:0] score,
    output reg  score_valid
);
    reg [{bits - 1}:0] shiftreg;
    always @(posedge clk) begin
        if (!rst_n) shiftreg <= 0;
        else if (load_en) shiftreg <= {{shiftreg[{bits - 2}:0], load_bit}};
    end

    reg [{iw - 1}:0] widx;
    reg [{cw - 1}:0] cidx;
    reg signed [{bias_w - 1}:0] acc;
    reg busy;

    wire [1:0] wsel = shiftreg[2*(cidx*{n_feat} + widx) +: 2];
    wire fbit = features[widx];

    always @(posedge clk) begin
        if (!rst_n) begin
            widx <= 0; cidx <= 0; acc <= 0; busy <= 0;
            score <= 0; score_valid <= 0;
        end else if (start) begin
            widx <= 0; cidx <= 0; busy <= 1; score_valid <= 0;
            acc <= $signed(shiftreg[2*{n_w} +: {bias_w}]);
        end else if (busy) begin
            if (fbit) begin
                if (wsel == 2'b01) acc <= acc + 1;
                else if (wsel == 2'b10) acc <= acc - 1;
            end
            if (widx == {n_feat - 1}) begin
                widx <= 0; score <= acc; score_valid <= 1;
                if (cidx == {n_class - 1}) busy <= 0;
                else begin
                    cidx <= cidx + 1;
                    acc <= $signed(shiftreg[2*{n_w} + {bias_w}*(cidx+1) +: {bias_w}]);
                end
            end else begin
                widx <= widx + 1; score_valid <= 0;
            end
        end else score_valid <= 0;
    end
endmodule
"""


# -------------------------------------------------------------- yosys

YS = """
read_verilog {src}
hierarchy -top {top}
flatten
proc; opt; fsm; opt; memory; opt
techmap; opt
dfflibmap -liberty {lib}
abc -liberty {lib}
opt_clean -purge
stat -liberty {lib}
"""


def run_yosys(verilog, top, lib):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "d.v").write_text(verilog)
        (td / "r.ys").write_text(YS.format(src=td / "d.v", top=top, lib=lib))
        r = subprocess.run(
            ["yosys", "-s", str(td / "r.ys")], capture_output=True, text=True
        )
        if r.returncode:
            print(r.stdout[-2500:], file=sys.stderr)
            raise SystemExit(f"yosys failed on {top}")
        cells = re.search(r"Number of cells:\s+(\d+)", r.stdout)
        area = re.search(r"Chip area for.*?:\s+([\d.]+)", r.stdout)
        return (
            int(cells.group(1)) if cells else 0,
            float(area.group(1)) if area else 0.0,
        )


def find_liberty(explicit=None):
    import os

    if explicit:
        return explicit
    for p in [
        os.environ.get("PDK_ROOT"),
        "/opt/pdk",
        "/foss/pdks",
        str(Path.home() / ".volare"),
    ]:
        if p and Path(p).exists():
            hits = sorted(Path(p).rglob("sg13g2_stdcell_typ*.lib"))
            if hits:
                return str(hits[0])
    if LIB_CACHE.exists():
        return str(LIB_CACHE)
    print("downloading liberty file (~1.7 MB) ...")
    import urllib.request

    urllib.request.urlretrieve(LIB_URL, LIB_CACHE)
    return str(LIB_CACHE)


# --------------------------------------------------------------- main

TILES = {
    "1x2": 36000,
    "2x2": 72000,
    "3x2": 108000,
    "4x2": 144000,
    "6x2": 216000,
    "8x2": 288000,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="trained weights")
    ap.add_argument("--lanes", type=int, nargs="+", default=None)
    ap.add_argument("--classes", type=int, default=6)
    ap.add_argument("--liberty", default=None)
    args = ap.parse_args()

    d = np.load(args.npz)
    W = d["W1"].astype(int)
    if W.shape[1] != PIXELS:
        W = W.T
    n_feat = W.shape[0]
    biases = d["b1"] if "b1" in d else np.zeros(n_feat, dtype=int)
    zero_rate = float((W == 0).mean())

    print(
        f"{args.npz}: {n_feat} features, {zero_rate:.1%} zeros, "
        f"accuracy {float(d['accuracy']):.4f}"
        if "accuracy" in d
        else f"{args.npz}: {n_feat} features, {zero_rate:.1%} zeros"
    )

    lanes = args.lanes or [
        l for l in (7, 8, 14, 16, 28, 32, n_feat) if l <= n_feat and n_feat % l == 0
    ]
    lib = find_liberty(args.liberty)

    hd_c, hd_a = run_yosys(emit_head(n_feat, args.classes), "head", lib)
    print(
        f"\nhead ({n_feat}->{args.classes}, loadable): "
        f"{hd_a:,.0f} um2  ({hd_a / NAND2_UM2:,.0f} GE)\n"
    )

    print(
        f"{'lanes':>6} {'passes':>7} {'cycles':>7} {'backbone um2':>13} "
        f"{'total um2':>11} {'tiles':>7}"
    )
    print("-" * 60)

    rows = []
    for L in lanes:
        v, passes = emit_serial_backbone(W, biases, L)
        c, a = run_yosys(v, "backbone", lib)
        # a backbone that folded to almost nothing is a bug, not a win
        floor = (n_feat + L * ACC_W) * NAND2_UM2
        flag = "  <-- suspect, too small" if a < floor else ""
        total = a + hd_a
        cycles = PIXELS * passes + n_feat * args.classes
        fit = next((n for n, cap in TILES.items() if total < 0.70 * cap), ">8x2")
        if not flag:
            rows.append((L, total))
        print(
            f"{L:>6} {passes:>7} {cycles:>7} {a:>13,.0f} {total:>11,.0f} {fit:>7}{flag}"
        )

    best = min(rows, key=lambda r: r[1])
    print(f"\nsmallest: {best[0]} lanes at {best[1]:,.0f} um2")
    print("""
Tile capacity (cell area, sg13g2):
   1x2  36,000    2x2  72,000    3x2 108,000    4x2 144,000

The 'tiles' column applies a 70% target, leaving headroom for routing and
buffering. Confirm with a real LibreLane run before committing.
""")


if __name__ == "__main__":
    main()
