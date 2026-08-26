"""K1 report generator: figures + README generated blocks, all from archived data.

Modes:
  python analysis/report/make_report.py           # write figures + regenerate
                                                  # README generated blocks and
                                                  # docs/k1/results_block64.md
  python analysis/report/make_report.py --check   # regenerate into memory,
                                                  # exit 1 if the committed
                                                  # blocks are stale

This script is the ONLY source of performance numbers in README's Results
section. README carries marker pairs

  <!-- GEN:name --> ... <!-- /GEN:name -->

and the script replaces the content between each pair. Every generated block
ends with a "Source: <file path(s)>" line naming the archived files it was
computed from. Before writing anything, the script re-derives a set of sanity
anchors from the archived files and refuses to run if any of them is off by
more than rounding.
"""

import argparse
import csv
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MATRIX_DIR = os.path.join(ROOT, "bench", "results", "matrix_20260824_213534")
MATRIX_REL = "bench/results/matrix_20260824_213534"
V0_JSON = os.path.join(
    ROOT, "bench", "results", "v0_32x8192x32x8x16_20260824_192119.json"
)
V0_REL = "bench/results/v0_32x8192x32x8x16_20260824_192119.json"
NCU_DIR = os.path.join(ROOT, "analysis", "ncu")
README = os.path.join(ROOT, "README.md")
BLOCK64_MD = os.path.join(ROOT, "docs", "k1", "results_block64.md")
FIG_DIR = os.path.join(ROOT, "docs", "k1", "figures")

HEADTYPES = [32, 64]  # Hq; Hkv is 8 throughout
S_LIST = [512, 2048, 8192, 32768]
B_LIST = [1, 8, 32, 128]
SPLITS_LIST = [1, 4, 8, 16, 32, 64]
REP = (32, 8192, 32, 16)  # (Hq, S, B, block): the representative cell
STARVE = (32, 32768, 1, 16)  # the starvation cell used in figure 3


# ---------------------------------------------------------------- data loading

def load_cells():
    """(Hq, S, B, block) -> cell dict from the archived matrix JSONs."""
    cells = {}
    for fname in sorted(os.listdir(MATRIX_DIR)):
        if not (fname.startswith("cell_") and fname.endswith(".json")):
            continue
        with open(os.path.join(MATRIX_DIR, fname)) as f:
            c = json.load(f)
        cells[(c["Hq"], c["S"], c["B"], c["block"])] = c
    if len(cells) != 64:
        raise SystemExit(f"expected 64 matrix cells, found {len(cells)}")
    return cells


def load_v0():
    with open(V0_JSON) as f:
        return json.load(f)


def load_ncu_rep(name):
    """Single-row NCU extraction CSV -> dict."""
    with open(os.path.join(NCU_DIR, f"{name}_rep.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, name
    return rows[0]


def load_ncu_cells():
    with open(os.path.join(NCU_DIR, "matrix_cells.csv")) as f:
        return list(csv.DictReader(f))


def splits_ms(cell, ns):
    return cell["results"]["v2"]["by_splits"][str(ns)]


# ----------------------------------------------------------------- formatting

def f_ms(x):
    return f"{x:.4f}"


def f_pct(x):
    return f"{x:.2f}"


def f_ratio(x):
    return f"{x:.2f}" if x >= 10 else f"{x:.3f}"


def f_bytes(b):
    if b >= 2**30:
        return f"{b / 2**30:g} GiB"
    return f"{b / 2**20:g} MiB"


def cell_label(hq, s, b, blk):
    return f"{hq}x8, S={s}, B={b}, block={blk}"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


# ------------------------------------------------------------ derived numbers

def block_effect(cells):
    """|delta| % between block 16 and 64 medians, relative to the faster of
    the two, over every (cell, kernel) pair present at both block sizes."""
    deltas = []
    for (hq, s, b, blk), c in cells.items():
        if blk != 16:
            continue
        c64 = cells[(hq, s, b, 64)]
        for k in c["results"]:
            if k not in c64["results"]:
                continue
            a = c["results"][k]["median_ms"]
            d = c64["results"][k]["median_ms"]
            deltas.append((abs(a - d) / min(a, d) * 100, hq, s, b, k))
    deltas.sort()
    n = len(deltas)
    median = (deltas[n // 2][0] if n % 2 else
              (deltas[n // 2 - 1][0] + deltas[n // 2][0]) / 2)
    over2 = [d for d in deltas if d[0] > 2.0]
    return n, median, over2, deltas[-1]


def v2_over_v1(cells):
    rats = [(c["results"]["v1"]["median_ms"] / c["results"]["v2"]["median_ms"],
             hq, s, b, blk)
            for (hq, s, b, blk), c in cells.items()]
    return min(rats), max(rats)


def ratio_extremes(cells, hq):
    rats = [(c["results"]["flashinfer"]["median_ms"]
             / c["results"]["v2"]["median_ms"], h, s, b, blk)
            for (h, s, b, blk), c in cells.items() if h == hq]
    return min(rats), max(rats)


def best_v2_pct(cells, hq):
    best = None
    for (h, s, b, blk), c in cells.items():
        if h != hq:
            continue
        v2 = c["results"]["v2"]
        if best is None or v2["pct_of_sol"] > best[0]:
            best = (v2["pct_of_sol"], s, b, blk, v2["best_splits"])
    return best


# -------------------------------------------------------------- sanity checks

def check(cond, what):
    if not cond:
        raise SystemExit(f"SANITY ANCHOR FAILED: {what} — refusing to publish. "
                         "Re-derive the anchor from the archived files and "
                         "resolve the discrepancy first.")


def sanity(cells, v0):
    rep = cells[REP]
    r = rep["results"]
    check(abs(r["v1"]["median_ms"] - 2.0044) < 5e-4, "rep v1 median 2.0044 ms")
    check(abs(r["v1"]["pct_of_sol"] - 15.99) < 5e-2, "rep v1 15.99 % of SoL")
    check(r["v2"]["best_splits"] == 32, "rep v2 best splits = 32")
    check(abs(r["v2"]["median_ms"] - 1.1426) < 5e-4, "rep v2 best 1.1426 ms")
    check(abs(r["v2"]["pct_of_sol"] - 28.05) < 5e-2, "rep v2 best 28.05 %")
    check(abs(splits_ms(rep, 8) - 1.1996) < 5e-4, "rep v2 splits=8 1.1996 ms")
    check(splits_ms(rep, 64) > splits_ms(rep, 32),
          "rep rollover: splits=64 slower than splits=32")
    check(abs(r["flashinfer"]["median_ms"] - 0.3682) < 5e-4,
          "rep FlashInfer 0.3682 ms")
    check(abs(r["flashinfer"]["pct_of_sol"] - 87.04) < 5e-2,
          "rep FlashInfer 87.04 %")
    check(abs(v0["median_ms"] - 7.8775) < 5e-4, "v0 single-shape 7.8775 ms")
    check(abs(v0["pct_of_sol"] - 4.07) < 5e-2, "v0 single-shape 4.07 %")

    for hq, want in [(32, 30.42), (64, 15.59)]:
        pct, s, b, blk, ns = best_v2_pct(cells, hq)
        check(abs(pct - want) < 5e-2 and (s, b, blk, ns) == (32768, 128, 64, 32),
              f"{hq}x8 best v2 %SoL {want} at S32768/B128/bs64/splits32")

    lo, hi = v2_over_v1(cells)
    check(abs(lo[0] - 0.983) < 5e-3 and lo[1:] == (64, 512, 128, 16),
          "v2/v1 min 0.983 at 64x8/S512/B128/bs16")
    check(abs(hi[0] - 38.60) < 5e-2 and hi[1:] == (32, 32768, 1, 64),
          "v2/v1 max 38.60 at 32x8/S32768/B1/bs64")

    n, median, over2, worst = block_effect(cells)
    check(n == 104, "block-effect pair count 104")
    check(abs(median - 0.49) < 5e-2, "block-effect median 0.49 %")
    check(len(over2) == 8, "8 block-effect pairs > 2 %")
    check(abs(worst[0] - 30.7) < 0.1 and worst[1:] == (32, 8192, 1, "v1"),
          "block-effect worst 30.7 % at 32x8/S8192/B1 v1")

    fi_max = max(c["results"]["flashinfer"]["pct_of_sol"]
                 for c in cells.values())
    check(abs(fi_max - 93.5) < 5e-2, "FlashInfer max %SoL 93.5")


# ------------------------------------------------------------------- figures

def make_figures(cells, v0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    rep = cells[REP]
    r = rep["results"]

    # -- figure 1: version ladder at the representative shape
    labels = ["v0", "v1", "v2\n(best splits=%d)" % r["v2"]["best_splits"],
              "FlashInfer\n0.6.17"]
    ms = [v0["median_ms"], r["v1"]["median_ms"], r["v2"]["median_ms"],
          r["flashinfer"]["median_ms"]]
    pct = [v0["pct_of_sol"], r["v1"]["pct_of_sol"], r["v2"]["pct_of_sol"],
           r["flashinfer"]["pct_of_sol"]]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bars = ax.bar(labels, pct, color=["C0", "C0", "C0", "C1"])
    for bar, m, p in zip(bars, ms, pct):
        ax.annotate(f"{m:.4f} ms\n{p:.2f} %",
                    (bar.get_x() + bar.get_width() / 2, p),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9)
    for i, (a, b) in enumerate([(0, 1), (1, 2)]):
        ax.annotate(f"{ms[a] / ms[b]:.2f}x",
                    ((a + b) / 2, max(pct[a], pct[b]) + 12),
                    ha="center", fontsize=10, color="C3")
        ax.annotate("", xy=(b, pct[b] + 10), xytext=(a, pct[a] + 10),
                    arrowprops=dict(arrowstyle="->", color="C3"))
    ax.set_ylabel("% of KV-bytes speed-of-light")
    ax.set_ylim(0, 105)
    ax.set_title("K1 ladder — 32Q/8KV, d=128, S=8192, B=32, block=16")
    fig.text(0.5, 0.01,
             "v0 from a separate single-shape run under the same timing "
             "contract; v1/v2/FlashInfer from the matrix cell.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(FIG_DIR, "k1_ladder.png"), dpi=150)
    plt.close(fig)

    # -- figure 2: % of SoL across the matrix, block=16
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, hq in zip(axes, HEADTYPES):
        for i, b in enumerate(B_LIST):
            v2 = [cells[(hq, s, b, 16)]["results"]["v2"]["pct_of_sol"]
                  for s in S_LIST]
            fi = [cells[(hq, s, b, 16)]["results"]["flashinfer"]["pct_of_sol"]
                  for s in S_LIST]
            ax.plot(S_LIST, v2, "o-", color=f"C{i}")
            ax.plot(S_LIST, fi, "s--", color=f"C{i}", alpha=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(S_LIST, [str(s) for s in S_LIST])
        ax.set_xlabel("S (uniform seq_len)")
        ax.set_title(f"{hq}Q/8KV")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("% of KV-bytes speed-of-light")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=f"C{i}", marker="o", label=f"B={b}")
               for i, b in enumerate(B_LIST)]
    handles += [Line2D([], [], color="k", marker="o", ls="-",
                       label="v2 (best splits)"),
                Line2D([], [], color="k", marker="s", ls="--",
                       label="FlashInfer")]
    axes[0].legend(handles=handles, fontsize=8, ncol=2)
    fig.suptitle("block=16; v2 at per-cell best num_splits", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "k1_matrix_sol.png"), dpi=150)
    plt.close(fig)

    # -- figure 3: splits sweep, speedup vs own num_splits=1
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, label, color in [
        (REP, "representative: 32x8, S=8192, B=32", "C0"),
        (STARVE, "starvation: 32x8, S=32768, B=1", "C1"),
    ]:
        c = cells[key]
        base = splits_ms(c, 1)
        speedup = [base / splits_ms(c, ns) for ns in SPLITS_LIST]
        ax.plot(SPLITS_LIST, speedup, "o-", color=color, label=label)
        best = c["results"]["v2"]["best_splits"]
        best_sp = base / splits_ms(c, best)
        ax.plot([best], [best_sp], "*", color=color, markersize=16)
        ax.annotate(f"best={best}", (best, best_sp),
                    textcoords="offset points", xytext=(6, 6), fontsize=9,
                    color=color)
    ax.set_xscale("log", base=2)
    ax.set_xticks(SPLITS_LIST, [str(n) for n in SPLITS_LIST])
    ax.set_xlabel("num_splits")
    ax.set_ylabel("speedup vs the same cell's num_splits=1")
    ax.set_title("v2 split-KV sweep (block=16)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "k1_splits_sweep.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------- generated blocks

def gen_headline(cells, v0):
    rep = cells[REP]
    r = rep["results"]
    lines = []
    lines.append(
        f"- Representative shape (32Q/8KV, d=128, S=8192, B=32, block=16, "
        f"{f_bytes(rep['kv_bytes'])} of KV): "
        f"v0 {f_ms(v0['median_ms'])} ms ({f_pct(v0['pct_of_sol'])} % of SoL) "
        f"→ v1 {f_ms(r['v1']['median_ms'])} ms "
        f"({f_pct(r['v1']['pct_of_sol'])} %) "
        f"→ v2 {f_ms(r['v2']['median_ms'])} ms "
        f"({f_pct(r['v2']['pct_of_sol'])} %, best num_splits="
        f"{r['v2']['best_splits']}); FlashInfer "
        f"{f_ms(r['flashinfer']['median_ms'])} ms "
        f"({f_pct(r['flashinfer']['pct_of_sol'])} %)."
    )
    parts = []
    for hq in HEADTYPES:
        pct, s, b, blk, ns = best_v2_pct(cells, hq)
        parts.append(f"{f_pct(pct)} % for {hq}Q/8KV "
                     f"(S={s}, B={b}, block={blk}, num_splits={ns})")
    lines.append(f"- Best v2 % of SoL over the 64-cell matrix: "
                 f"{parts[0]} and {parts[1]}.")
    parts = []
    for hq in HEADTYPES:
        lo, hi = ratio_extremes(cells, hq)
        parts.append(
            f"{hq}Q/8KV {f_ratio(lo[0])} ({cell_label(*lo[1:])}) to "
            f"{f_ratio(hi[0])} ({cell_label(*hi[1:])})"
        )
    lines.append(
        "- v2 ratio_vs_flashinfer (FlashInfer median / v2 median; > 1 would "
        f"mean faster than the baseline): {parts[0]}; {parts[1]}."
    )
    lines.append(f"\nSource: {MATRIX_REL}/ (cell JSONs), {V0_REL}")
    return "\n".join(lines)


def gen_ladder_table(cells, v0):
    rep = cells[REP]
    r = rep["results"]
    ncu = {k: load_ncu_rep(k) for k in ("v0", "v1", "v2")}

    def ncu_cols(row):
        stalls = ", ".join(row[f"stall_{i}"] for i in (1, 2, 3))
        return [f_pct(float(row["dram_throughput_pct_of_peak"])),
                f"{float(row['l1tex_global_load_sectors_per_request']):.2f}",
                f_pct(float(row["warps_active_pct_of_peak"])), stalls]

    sol = rep["sol_ms"]
    v2_8 = splits_ms(rep, 8)
    rows = [
        ["v0 (single-shape run)", f_ms(v0["median_ms"]),
         f_pct(v0["pct_of_sol"]), "—"] + ncu_cols(ncu["v0"]),
        ["v1", f_ms(r["v1"]["median_ms"]), f_pct(r["v1"]["pct_of_sol"]),
         f"{v0['median_ms'] / r['v1']['median_ms']:.2f}x vs v0"]
        + ncu_cols(ncu["v1"]),
        ["v2, splits=8 (default)", f_ms(v2_8), f_pct(sol / v2_8 * 100),
         f"{r['v1']['median_ms'] / v2_8:.2f}x vs v1"] + ncu_cols(ncu["v2"]),
        ["v2, best splits=%d" % r["v2"]["best_splits"],
         f_ms(r["v2"]["median_ms"]), f_pct(r["v2"]["pct_of_sol"]),
         f"{r['v1']['median_ms'] / r['v2']['median_ms']:.2f}x vs v1",
         "—", "—", "—", "—"],
        ["FlashInfer 0.6.17", f_ms(r["flashinfer"]["median_ms"]),
         f_pct(r["flashinfer"]["pct_of_sol"]), "—", "—", "—", "—", "—"],
    ]
    table = md_table(
        ["kernel", "median ms", "% of SoL", "speedup",
         "DRAM % (NCU)", "sectors/req", "occupancy %", "top-3 stalls (cyc/inst)"],
        rows,
    )
    footer = (
        "\nNCU columns are from separate single-launch profiles at this shape "
        "(the v2 NCU row is num_splits=8, matching the profiled "
        "configuration). NCU replays a single cold launch while bench medians "
        "come from warmed runs, so compare trends within one tool, not "
        "absolutes across tools (notes_w1.md). v0 timing comes from a "
        "separate single-shape run under the same timing contract; v0 is not "
        "in the matrix at this shape."
    )
    return (table + footer +
            f"\n\nSource: {MATRIX_REL}/cell_32x8_S8192_B32_bs16.json, "
            f"{V0_REL}, analysis/ncu/v0_rep.csv, analysis/ncu/v1_rep.csv, "
            f"analysis/ncu/v2_rep.csv")


def _grid_table(cells, hq, fmt_cell):
    header = ["S \\ B"] + [str(b) for b in B_LIST]
    rows = [[str(s)] + [fmt_cell(cells[(hq, s, b, 16)]) for b in B_LIST]
            for s in S_LIST]
    return md_table(header, rows)


def gen_ratio(cells, hq):
    def ratio_cell(c):
        r = c["results"]
        ratio = r["flashinfer"]["median_ms"] / r["v2"]["median_ms"]
        return f"{f_ratio(ratio)} ({r['v2']['best_splits']})"

    def fi_cell(c):
        return f_pct(c["results"]["flashinfer"]["pct_of_sol"])

    out = (f"v2 ratio_vs_flashinfer at block=16, {hq}Q/8KV — each cell is "
           "ratio (best num_splits):\n\n")
    out += _grid_table(cells, hq, ratio_cell)
    out += f"\n\nFlashInfer % of SoL at block=16, {hq}Q/8KV:\n\n"
    out += _grid_table(cells, hq, fi_cell)
    out += f"\n\nSource: {MATRIX_REL}/ (cell JSONs)"
    return out


def pershape_rows(cells, blk):
    rows = []
    for hq in HEADTYPES:
        for s in S_LIST:
            for b in B_LIST:
                c = cells[(hq, s, b, blk)]
                r = c["results"]
                ratio = (r["flashinfer"]["median_ms"]
                         / r["v2"]["median_ms"])
                if "v0" in r:
                    v0c = (f"{f_ms(r['v0']['median_ms'])} / "
                           f"{f_pct(r['v0']['pct_of_sol'])}")
                else:
                    v0c = "—"
                rows.append([
                    f"{hq}x8", s, b, f_bytes(c["kv_bytes"]),
                    f_ms(c["sol_ms"]),
                    f"{f_ms(r['flashinfer']['median_ms'])} / "
                    f"{f_pct(r['flashinfer']['pct_of_sol'])}",
                    v0c,
                    f"{f_ms(r['v1']['median_ms'])} / "
                    f"{f_pct(r['v1']['pct_of_sol'])}",
                    f"{f_ms(r['v2']['median_ms'])} / "
                    f"{f_pct(r['v2']['pct_of_sol'])} / "
                    f"{r['v2']['best_splits']}",
                    f_ratio(ratio),
                ])
    return rows


PERSHAPE_HEADER = ["heads", "S", "B", "KV bytes", "SoL ms",
                   "FlashInfer ms / %SoL", "v0 ms / %SoL", "v1 ms / %SoL",
                   "v2 ms / %SoL / best splits", "v2 ratio"]


def gen_pershape(cells):
    out = "All block=16 cells (both headtypes). "
    out += ("The block=64 table lives in "
            "[docs/k1/results_block64.md](docs/k1/results_block64.md). "
            "v0 was benchmarked only where it is not prohibitively slow "
            "(S ≤ 2048 and B ≤ 8).\n\n")
    out += md_table(PERSHAPE_HEADER, pershape_rows(cells, 16))
    out += f"\n\nSource: {MATRIX_REL}/ (cell JSONs)"
    return out


def render_block64(cells):
    out = "# K1 matrix results, block=64\n\n"
    out += ("Generated by `analysis/report/make_report.py` — do not edit by "
            "hand. Companion to the block=16 table in the "
            "[README](../../README.md); same columns, same run "
            "(matrix_20260824_213534), block=64 cells.\n\n")
    out += md_table(PERSHAPE_HEADER, pershape_rows(cells, 64))
    out += f"\n\nSource: {MATRIX_REL}/ (cell JSONs)\n"
    return out


def gen_block_effect(cells):
    n, median, over2, worst = block_effect(cells)
    all_b1 = all(d[3] == 1 for d in over2)
    out = (
        f"Across the {n} (cell, kernel) pairs present at both block sizes, "
        f"the median |Δ| between the block=16 and block=64 medians is "
        f"{median:.2f} % (Δ relative to the faster of the two). "
        f"{len(over2)} pairs exceed 2 %"
    )
    if all_b1:
        out += ", all of them B=1 cells"
    out += (f"; the worst is {worst[0]:.1f} % "
            f"({worst[1]}x8, S={worst[2]}, B={worst[3]}, {worst[4]}).")
    out += f"\n\nSource: {MATRIX_REL}/ (cell JSONs)"
    return out


def gen_v2_over_v1(cells):
    lo, hi = v2_over_v1(cells)
    out = (
        f"v2 speedup over v1 (v1 median / v2 best median) ranges from "
        f"{f_ratio(lo[0])}x ({cell_label(*lo[1:])}) to "
        f"{f_ratio(hi[0])}x ({cell_label(*hi[1:])})."
    )
    out += f"\n\nSource: {MATRIX_REL}/ (cell JSONs)"
    return out


def gen_argbest(cells):
    def argbest_cell(c):
        return str(c["results"]["v2"]["best_splits"])

    out = "Best num_splits per cell (block=16):\n\n"
    for hq in HEADTYPES:
        out += f"{hq}Q/8KV:\n\n"
        out += _grid_table(cells, hq, argbest_cell) + "\n\n"

    below, slower = 0, 0
    for hq in HEADTYPES:
        for s in S_LIST:
            for b in B_LIST:
                c = cells[(hq, s, b, 16)]
                best = c["results"]["v2"]["best_splits"]
                if best < 64:
                    below += 1
                    if splits_ms(c, 64) > splits_ms(c, best):
                        slower += 1
    rep = cells[REP]
    out += (
        f"Rollover check over these 32 block=16 cells: {below} cells have "
        f"argbest < 64, and in {slower} of those {below} splits=64 is "
        f"strictly slower than argbest — the rollover the W1 notes deferred "
        f"is observed. At the representative cell the sweep peaks at "
        f"num_splits=32 ({f_ms(splits_ms(rep, 32))} ms) and 64 is slower "
        f"({f_ms(splits_ms(rep, 64))} ms)."
    )
    out += f"\n\nSource: {MATRIX_REL}/ (cell JSONs, by_splits)"
    return out


def gen_ncu_cells(cells):
    rows_in = load_ncu_cells()
    rows = []
    for r in rows_in:
        rows.append([
            r["cell"],
            f_pct(float(r["dram_throughput_pct_of_peak"])),
            f"{float(r['l1tex_global_load_sectors_per_request']):.2f}",
            f_pct(float(r["warps_active_pct_of_peak"])),
            f_pct(float(r["l2_hit_rate_pct"])),
            f"{float(r['dram_bytes_sum']):.4g}",
            f"{float(r['lts_bytes_sum']):.4g}",
            f"{float(r['merge_kernel_ms']):.4f}",
        ])
    table = md_table(
        ["cell", "DRAM %", "sectors/req", "occupancy %", "L2 hit %",
         "DRAM bytes", "L2 bytes", "merge ms"],
        rows,
    )
    by = {r["cell"].split(":")[0]: r for r in rows_in}
    dram_ba = (float(by["B"]["dram_bytes_sum"])
               / float(by["A"]["dram_bytes_sum"]))
    lts_ba = float(by["B"]["lts_bytes_sum"]) / float(by["A"]["lts_bytes_sum"])
    # C's partial/merge durations come from the same NCU session (cold
    # single launch), so the share compares like with like.
    c_partial_us, c_merge_us = None, None
    with open(os.path.join(NCU_DIR, "cellC_rep_raw.csv")) as f:
        for r in csv.DictReader(f):
            dur = r.get("gpu__time_duration.sum", "")
            if not dur or dur == "us":
                continue  # units row
            if "partial_kernel" in r["Kernel Name"]:
                c_partial_us = float(dur)
            elif "merge_kernel" in r["Kernel Name"]:
                c_merge_us = float(dur)
    assert c_partial_us and c_merge_us, "cellC durations not found"
    derived = (
        f"\nDerived: DRAM bytes B/A = {dram_ba:.3f}x; L2 (lts) bytes B/A = "
        f"{lts_ba:.2f}x; in C the merge kernel is {c_merge_us:.1f} µs on a "
        f"{c_partial_us:.1f} µs partial (both NCU gpu__time_duration from "
        f"the same profile), i.e. "
        f"~{c_merge_us / c_partial_us * 100:.0f} % of the partial-kernel "
        f"time."
    )
    return (table + derived +
            "\n\nSource: analysis/ncu/matrix_cells.csv, "
            "analysis/ncu/cellC_rep_raw.csv")


def build_blocks(cells, v0):
    return {
        "headline": gen_headline(cells, v0),
        "ladder_table": gen_ladder_table(cells, v0),
        "ratio_32x8": gen_ratio(cells, 32),
        "ratio_64x8": gen_ratio(cells, 64),
        "pershape": gen_pershape(cells),
        "block_effect": gen_block_effect(cells),
        "v2_over_v1": gen_v2_over_v1(cells),
        "argbest": gen_argbest(cells),
        "ncu_cells": gen_ncu_cells(cells),
    }


# -------------------------------------------------------------- README edit

MARKER_RE = re.compile(
    r"(<!-- GEN:([a-z0-9_]+) -->)(.*?)(<!-- /GEN:\2 -->)", re.S
)


def apply_blocks(text, blocks):
    used = set()

    def sub(m):
        name = m.group(2)
        if name not in blocks:
            raise SystemExit(f"README has marker GEN:{name} but the "
                             "generator defines no such block")
        used.add(name)
        return f"{m.group(1)}\n{blocks[name].rstrip()}\n{m.group(4)}"

    out = MARKER_RE.sub(sub, text)
    missing = sorted(set(blocks) - used)
    if missing:
        raise SystemExit(f"README is missing markers for: {missing}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="regenerate into memory and exit 1 if the committed "
                         "README blocks or results_block64.md are stale")
    args = ap.parse_args()

    cells = load_cells()
    v0 = load_v0()
    sanity(cells, v0)

    blocks = build_blocks(cells, v0)
    with open(README) as f:
        readme_old = f.read()
    readme_new = apply_blocks(readme_old, blocks)
    block64_new = render_block64(cells)

    if args.check:
        stale = []
        if readme_new != readme_old:
            for m in MARKER_RE.finditer(readme_old):
                if m.group(3).strip() != blocks[m.group(2)].strip():
                    stale.append(f"README GEN:{m.group(2)}")
        try:
            with open(BLOCK64_MD) as f:
                if f.read() != block64_new:
                    stale.append("docs/k1/results_block64.md")
        except FileNotFoundError:
            stale.append("docs/k1/results_block64.md (missing)")
        if stale:
            print("STALE generated content:")
            for s in stale:
                print(f"  - {s}")
            sys.exit(1)
        print("check OK: all generated blocks match the archived data")
        return

    make_figures(cells, v0)
    with open(README, "w") as f:
        f.write(readme_new)
    with open(BLOCK64_MD, "w") as f:
        f.write(block64_new)
    print(f"wrote {len(blocks)} README blocks, docs/k1/results_block64.md, "
          f"and 3 figures under docs/k1/figures/")


if __name__ == "__main__":
    main()
