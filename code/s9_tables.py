# -*- coding: utf-8 -*-
"""Write booktabs LaTeX tables (figures/TABLE_*.tex) from results/ JSON. No number is typed by hand."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json

from common import RESULTS, ROOT, CONFIGS, load_json

FIG = os.path.join(ROOT, "figures")


def L(rel):
    p = os.path.join(RESULTS, rel)
    return load_json(p) if os.path.exists(p) else None


def pct(x, d=1):
    return "--" if x is None else f"{100 * x:.{d}f}"


def ci(c, d=1):
    return "" if not c else f" [{100 * c[0]:.{d}f}, {100 * c[1]:.{d}f}]"


def write(name, caption, label, colspec, header, rows, note=None, size="\\small", resize=None):
    wide = len([c for c in colspec if c in "lrcp"]) >= 7 if resize is None else resize
    lines = ["\\begin{table}[!htbp]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if wide:
        lines.append("\\resizebox{\\linewidth}{!}{%")
    lines += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(r if r.rstrip().endswith("\\\\") else r + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}" + ("}" if wide else "")]
    if note:
        lines.append(f"\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\footnotesize {note}}}")
    lines.append("\\end{table}")
    with open(os.path.join(FIG, name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", name)


def t_leakage():
    g = L("audit/grouping_sensitivity.json")
    rows = []
    for r in g["sensitivity"]:
        star = "$^{\\ast}$" if r["tau"] == 8 else ""
        rows.append(f"{r['tau']}{star} & {r['val_contaminated']} & {r['audit_images']} & {r['audit_groups']} & "
                    f"{r['counts']['head|small']} & {r['counts']['mid|small']} & {r['counts']['tail|small']} ({r['groups']['tail|small']})")
    h = g["val_nearest_train_phash_hist"]
    exact = h.get("0", 0)
    write("TABLE_leakage.tex",
          "Train/validation near-duplicate grouping and the resulting group-disjoint audit set (RicePest-30, 890 frame-consistent validation images).",
          "tab:leakage", "rrrrrrr",
          "$\\tau$ & Contaminated val & Audit images & Audit groups & Head-small & Mid-small & Tail-small (groups)",
          rows,
          note=f"$\\tau$: 64-bit pHash Hamming threshold for a near-duplicate link (links also require a shared class; byte-identical files are always linked). "
               f"$^{{\\ast}}$Frozen setting. {exact} of 933 validation images have a training image at pHash distance 0; "
               f"{g['exact_md5_cross_split_groups']} byte-identical (MD5) groups span both splits.")


def t_truncation():
    tr = L("audit/truncation.json")
    idx = {(r["T"], r["overlap"], r["set"], r["stratum"], r["size"]): r for r in tr}
    rows = []
    for T, ov in [(320, 0.25), (320, 0.1), (320, 0.0), (256, 0.25), (480, 0.25)]:
        def cell(s, st, z):
            r = idx.get((T, ov, s, st, z))
            return "--" if not r else f"{r['truncated']}/{r['n']}"
        a = idx.get((T, ov, "audit", "all", "small"))
        cut = pct(a["boundary_cut"] / a["n"]) if a else "--"
        star = "$^{\\ast}$" if (T, ov) == (320, 0.25) else ""
        rows.append(f"RicePest-30 & {T}/{int(T * (1 - ov))}{star} & {cell('train', 'all', 'small')} & {cell('val', 'all', 'small')} & "
                    f"{cell('audit', 'all', 'small')} & {cell('audit', 'tail', 'small')} & {cut}")
    y = L("yst/truncation.json")
    if y:
        yi = {(r["T"], r["overlap"], r["set"], r["cls"]): r for r in y}
        for T, ov in [(640, 0.25), (640, 0.1), (640, 0.0), (512, 0.25)]:
            a = yi[(T, ov, "all", "all")]
            per = ", ".join(f"{c} {yi[(T, ov, 'all', c)]['truncated']}" for c in ("WF", "MR", "NC"))
            star = "$^{\\ast}$" if (T, ov) == (640, 0.25) else ""
            rows.append(f"YST (native) & {T}/{int(T * (1 - ov))}{star} & \\multicolumn{{4}}{{c}}{{{a['truncated']}/{a['n']} all objects ({per})}} & {pct(a['boundary_cut'] / a['n'])}")
    write("TABLE_truncation.tex",
          "Literal truncation: objects with no full view in any SAHI tile (counts truncated/total). RicePest-30 rows count small objects ($<32^2$ in the 640-long-edge frame).",
          "tab:truncation", "llrrrrr",
          "Data & $T$/stride & Train small & Val small & Audit small & Audit tail-small & Seam-cut (\\%)",
          rows,
          note="Seam-cut: share of objects crossed by at least one tile border (they still have a full view in a neighbouring tile). "
               "$^{\\ast}$Frozen configuration. Overlap 0 is a positive control: it removes the redundancy that guarantees a full view.")


def part_row(lab, p, with_ci=False):
    n, k = p["n_objects"], p["n_misses"]
    c = p["counts"]
    if "prop_of_misses" not in p:
        return [f"{lab} & {n} ({p['n_groups']}) & {pct(p['recall'])} & {k} & {c['no_candidate']} & "
                f"{c['pre_suppression_only']} & {c['post_hoc_recoverable']}"]
    pm = p["prop_of_misses"]; cm = p.get("ci95_prop_of_misses", {})
    def f(cat):
        return f"{c[cat]} ({pct(pm[cat], 0)})"
    main = (f"{lab} & {n} ({p['n_groups']}) & {pct(p['recall'])} & {k} & {f('no_candidate')} & "
            f"{f('pre_suppression_only')} & {f('post_hoc_recoverable')}")
    if not with_ci or not cm:
        return [main]
    b = lambda x: f"[{100 * x[0]:.0f}, {100 * x[1]:.0f}]"
    rc = p.get("ci95_recall")
    return [main, f"\\quad 95\\% CI & & {b(rc) if rc else ''} & & {b(cm['no_candidate'])} & {b(cm['pre_suppression_only'])} & "
                  f"{b(cm['post_hoc_recoverable'])}"]


def t_partition(run="R3-ledger-yolo-T320", name="TABLE_partition.tex", det="YOLO11n"):
    s = L(f"stats/{run}_stats.json")
    if not s:
        return
    P = s["partitions"]
    order = [("Tail $<32^2$ (primary)", "tail|small"), ("\\quad Tail $<16^2$", "tail|tiny"), ("\\quad Tail $16^2$--$32^2$", "tail|s16_32"),
             ("Mid $<32^2$", "mid|small"), ("Head $<32^2$", "head|small"), ("Tail $\\geq 32^2$", "tail|notsmall"),
             ("Mid $\\geq 32^2$", "mid|notsmall"), ("Head $\\geq 32^2$", "head|notsmall"), ("All objects", "all|all")]
    rows = [r for lab, k in order if k in P and P[k]["n_objects"] > 0
            for r in part_row(lab, P[k], with_ci=k in ("tail|small", "head|small"))]
    con = s["size_matched_frequency_contrast"]
    note = (f"Group-disjoint audit set, frozen {det} + SAHI ($T$=320, overlap 0.25, GREEDYNMM/IOS 0.5 at the 0.01 floor), "
            "operating point score $\\geq$0.25 and 100 boxes per image. Miss columns: count (\\% of misses); CI rows: 95\\% cluster-bootstrap CI over audit groups; "
            "rates and CIs are omitted when a cell has $<$10 events or $<$5 groups. "
            f"Size-matched contrast (tail-small vs head-small standardized to the tail size mix): no-candidate share difference "
            f"{pct(con['diff']['no_candidate'])} pp{ci(con['ci95']['no_candidate'])}, Holm $p$={con['holm_p_3cats']['no_candidate']:.3f}; "
            f"pre-suppression-only {pct(con['diff']['pre_suppression_only'])} pp{ci(con['ci95']['pre_suppression_only'])}, Holm $p$={con['holm_p_3cats']['pre_suppression_only']:.3f}; "
            f"post-hoc-recoverable {pct(con['diff']['post_hoc_recoverable'])} pp{ci(con['ci95']['post_hoc_recoverable'])}, Holm $p$={con['holm_p_3cats']['post_hoc_recoverable']:.3f}.")
    write(name, f"Recovery-ceiling partition of missed objects ({det}).", "tab:partition" if name == "TABLE_partition.tex" else "tab:partition-" + det.lower(),
          "lrrrrrr", "Stratum, size & $N$ (groups) & Recall (\\%) & Misses & No cand. & Pre-supp. & Post-hoc",
          rows, note=note, size="\\footnotesize", resize=False)


NAMES = {"baseline": "SAHI baseline (frozen)", "norcal": "NorCal-style prior correction", "cls_size_calib": "Class$\\times$size thresholds",
         "na_agnostic": "Na-style rescoring, class-agnostic", "na_perclass": "Na-style rescoring, per class",
         "ml_greedynmm_calib": "Multi-label evidence + SAHI merge + thr.", "ccpr_std_evidence": "Peak read-out on std evidence + thr.",
         "ccpr_nocalib": "CCPR without class$\\times$size thr.", "ccpr": "CCPR (multi-label peak read-out + thr.)"}


def t_methods(run="R5R9-methods-yolo", name="TABLE_methods.tex", det="YOLO11n"):
    d = L(f"methods/{run}.json")
    if not d:
        return
    rows = []
    for m in NAMES:
        if m not in d["methods"]:
            continue
        o = d["methods"][m]["own"]; f = d["methods"][m]["matched_fp"]
        rec = d["attribution"].get(f"{m}|matched_fp", {}).get("recovered", {})
        lost = d["attribution"].get(f"{m}|matched_fp", {}).get("lost", {})
        ts = f"{o['tp|tail|small']}/{o['n|tail|small']}"
        tsf = f"{f['tp|tail|small']}/{f['n|tail|small']}"
        rows.append(f"{NAMES[m]} & {pct(o['recall|all'])} & {ts} & {pct(o['precision|head'])} & {o['FP']} & "
                    f"{pct(f['recall|all'])} & {pct(f['recall|head|small'])} & {tsf} & "
                    f"{rec.get('pre_suppression_only', 0)}/{rec.get('post_hoc_recoverable', 0)}/{rec.get('no_candidate', 0)} & {lost.get('all', 0)}")
    write(name, f"Same-dump post-processing methods on the audit set ({det}; parameters fit by 5-fold group cross-fitting).",
          "tab:methods" if name == "TABLE_methods.tex" else "tab:methods-" + det.lower(), "lrrrrrrrrr",
          "Method & \\multicolumn{4}{c}{Own operating point} & \\multicolumn{5}{c}{Matched false positives} \\\\ "
          "\\cmidrule(lr){2-5}\\cmidrule(lr){6-10} & R all & Tail-small & P head & FP & R all & R head-small & Tail-small & Recovered P/H/N & Lost",
          rows, size="\\scriptsize",
          note="R: recall@0.5 (\\%); P head: precision of emitted head-class boxes (\\%); tail-small: detected/total. "
               "Matched FP: one global threshold per method chosen so that its total false positives do not exceed the baseline's. "
               "Recovered P/H/N: baseline misses recovered at matched FP, by the baseline's recovery-ceiling category "
               "(pre-suppression only / post-hoc recoverable / no candidate); Lost: baseline detections no longer detected.")


def t_ccpr(run="R5R9-methods-yolo"):
    d = L(f"methods/{run}.json")
    if not d:
        return
    rows = []
    lab = {"ccpr[own]_vs_calib[own]": "Own operating points", "ccpr[matched_head_precision]_vs_calib[own]": "Matched head precision",
           "ccpr[matched_fp]_vs_calib[matched_fp]": "Matched false positives"}
    for k, g in d["ccpr_go_no_go"].items():
        t, h, c = g["tail_small"], g["head"], g["criteria"]
        yn = lambda b: "yes" if b else "no"
        rows.append(f"{lab.get(k, k)} & {t['extra_tp']:+d} & {100 * t['diff']:+.1f}{ci(t['ci95'])} & "
                    f"{100 * h['diff']:+.1f} ({100 * h['lower_one_sided_95']:+.1f}) & {100 * g['NI_margin']:.0f} & "
                    f"{yn(c['extra_tail_small_tp>=5'])}/{yn(c['head_NI_pass'])}/{yn(c['tail_small_CI_excludes_0'])} & {'GO' if g['GO'] else 'NO-GO'}")
    write("TABLE_ccpr.tex", "Pre-registered go/no-go test of CCPR against class$\\times$size threshold calibration (audit set, YOLO11n).",
          "tab:ccpr", "lrrrrrr",
          "Comparison & Extra tail-small TP & $\\Delta$ tail-small recall, pp [95\\% CI] & $\\Delta$ head recall, pp (one-sided 95\\% LB) & NI margin (pp) & Criteria 1/2/3 & Decision",
          rows, size="\\scriptsize",
          note="Criteria: (1) at least 5 additional tail-small true positives; (2) head non-inferiority, one-sided paired cluster bootstrap lower bound above $-$margin; "
               "(3) 95\\% cluster-bootstrap CI of the tail-small recall difference excludes 0. All three must hold for GO.")


def summ_row(lab, run):
    d = L(f"ledger/{run}.json")
    if not d:
        return None
    a = d["summary"]["audit"]
    def r(k):
        t = a.get(k, {}); return t.get("D", 0) / t["n"] if t.get("n") else None
    sm = a.get("all|small", {}); ts = a.get("tail|small", {})
    miss = sm.get("n", 0) - sm.get("D", 0)
    sh = lambda c: pct(sm.get("cat_" + c, 0) / miss, 0) if miss else "--"
    tsm = f"{ts.get('cat_no_candidate', 0)}/{ts.get('cat_pre_suppression_only', 0)}/{ts.get('cat_post_hoc_recoverable', 0)}"
    prec = d["summary"]["audit_precision"].get("all", {}).get("precision")
    return (f"{lab} & {pct(r('all|all'))} & {pct(prec)} & {pct(r('all|small'))} & {ts.get('D', 0)}/{ts.get('n', 0)} & "
            f"{sh('no_candidate')}/{sh('pre_suppression_only')}/{sh('post_hoc_recoverable')} & {tsm}")


def t_sensitivity():
    groups = [
        ("Primary", [("YOLO11n, $T$=320/0.25, GREEDYNMM-IOS, $\\tau$=0.25, cap 100", "R3-ledger-yolo-T320")]),
        ("Read-out", [("NMS-IoU 0.5 (class-aware)", "S-ledger-yolo-T320-nms"), ("GREEDYNMM-IOS, class-agnostic", "S-ledger-yolo-T320-agn"),
                      ("Threshold before merge (SAHI deployment order)", "S-ledger-yolo-T320-prethr"),
                      ("Full image only (no slicing)", "S-ledger-yolo-fullimage")]),
        ("Tiling", [("$T$=256, overlap 0.25", "S-ledger-yolo-T256"), ("$T$=480, overlap 0.25", "S-ledger-yolo-T480"),
                    ("$T$=320, overlap 0 (positive control)", "S-ledger-yolo-T320-ov0")]),
        ("Operating point", [("$\\tau$=0.10", "S-ledger-yolo-T320-tau0.1"), ("$\\tau$=0.40", "S-ledger-yolo-T320-tau0.4"),
                             ("Global cap 20", "S-ledger-yolo-T320-cap20"), ("Global cap 300", "S-ledger-yolo-T320-cap300"),
                             ("Per-class cap 5", "S-ledger-yolo-T320-pcc5"), ("Per-class cap 20", "S-ledger-yolo-T320-pcc20")]),
        ("Detector", [("FCOS-R50-FPN, primary pipeline", "R7-ledger-fcos-T320"),
                      ("FCOS-R50-FPN, NMS-IoU 0.5", "S-ledger-fcos-T320-nms_iou"),
                      ("FCOS-R50-FPN, threshold before merge", "S-ledger-fcos-T320-greedynmm_ios_prethr")]),
    ]
    rows = []
    for gname, items in groups:
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{gname}}}}} \\\\")
        for lab, run in items:
            r = summ_row(lab, run)
            if r:
                rows.append(r)
    write("TABLE_sensitivity.tex", "Sensitivity of recall and of the recovery-ceiling partition to the read-out, tiling, operating point and detector (audit set).",
          "tab:sensitivity", "lrrrrrr",
          "Configuration & R all & P all & R small & Tail-small det. & Small misses N/P/H (\\%) & Tail-small misses N/P/H",
          rows, size="\\scriptsize",
          note="N/P/H: no candidate / pre-suppression only / post-hoc recoverable. R, P in \\%. Tail-small det.: detected/total.")


def t_yst():
    d = L("ledger/R8-ledger-yst-T640.json")
    y = L("yst/truncation.json")
    if not d:
        return
    a = d["summary"]["audit"]
    yi = {(r["T"], r["overlap"], r["set"], r["cls"]): r for r in (y or [])}
    d2 = L("ledger/S-ledger-yst-T640-nms-cap300.json")
    b = d2["summary"]["audit"] if d2 else {}
    rows = []
    for c in ("WF", "MR", "NC", "all"):
        t = a.get(f"{c}|all", {})
        n = t.get("n", 0)
        if not n:
            continue
        tr = yi.get((640, 0.25, "test", c))
        u = b.get(f"{c}|all", {})
        nph = lambda q: f"{q.get('cat_no_candidate', 0)}/{q.get('cat_pre_suppression_only', 0)}/{q.get('cat_post_hoc_recoverable', 0)}"
        rows.append(f"{c if c != 'all' else 'All'} & {n} & {(str(tr['truncated']) + '/' + str(tr['n'])) if tr else '--'} & "
                    f"{pct(t.get('D', 0) / n)} & {nph(t)} & "
                    f"{pct(u.get('D', 0) / n) if u else '--'} & {nph(u) if u else '--'}")
    prec = d["summary"]["audit_precision"].get("all", {}).get("precision")
    write("TABLE_yst.tex", "Native-resolution source re-check on Yellow Sticky Traps (held-out test images, YOLO11n trained on native 640-pixel tiles).",
          "tab:yst", "lrrrrrr",
          "Class & $N$ & No full view & \\multicolumn{2}{c}{Frozen pipeline} & \\multicolumn{2}{c}{NMS read-out, cap 300} \\\\ "
          "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7} & & & Recall (\\%) & Misses N/P/H & Recall (\\%) & Misses N/P/H",
          rows, size="\\scriptsize",
          note=f"Native 5184$\\times$3456 images, SAHI $T$=640, overlap 0.25 plus the full-image view; frozen pipeline = same read-out and operating point as RicePest-30. N/P/H: no candidate / pre-suppression only / post-hoc recoverable. "
               f"Precision of all emitted boxes: {pct(prec)}\\%. WF: \\textit{{Trialeurodes vaporariorum}} (whitefly); MR: \\textit{{Macrolophus pygmaeus}}; NC: \\textit{{Nesidiocoris tenuis}}. Three classes, 8.4$\\times$ imbalance: not a long-tail test.")


def t_config():
    c = load_json(os.path.join(CONFIGS, "prereg_frozen.json"))
    ya = L("runs/yolo11n_rice_s0_train.json") or {}
    fc = L("runs/fcos_rice_s0_train.json") or {}
    rows = [f"Detector A & YOLO11n, COCO init, 640 input, {c['detector_A']['epochs']} epochs, batch 16, seed 0, last checkpoint (no selection on validation)",
            f"Detector B & FCOS-ResNet50-FPN (torchvision), COCO init, 640, {fc.get('epochs', c['detector_B'].get('epochs', 6))} epochs, batch {fc.get('batch', 4)}, last checkpoint",
            "Reference frame & image resized to 640-pixel long edge; size bins on box area in this frame",
            "Views & full image + SAHI tiles $T$=320, overlap 0.25 (stride 240); every view letterboxed to 640",
            "Candidate dump & per-view score floor 0.01, per-view NMS IoU 0.7, max 300 boxes; single label per anchor",
            "Read-out (merge) & SAHI GREEDYNMM, IOS 0.5, class-aware, on the floor-level dump",
            "Operating point & score $\\geq$0.25, at most 100 boxes per image; matching IoU $\\geq$0.5, one-to-one",
            "Grouping & 64-bit pHash, Hamming $\\leq$8 with a shared class, or identical MD5; connected components",
            "Audit set & validation images whose component holds no training image",
            "Statistics & cluster percentile bootstrap over audit groups (2,000), Holm over 3 categories, class-label permutation (10,000)"]
    if ya.get("frozen_sha256"):
        rows.append(f"Weights hash & YOLO11n {ya['frozen_sha256'][:12]}; FCOS {fc.get('frozen_sha256', '--')[:12]}")
    write("TABLE_config.tex", "Frozen configuration (written to \\texttt{configs/prereg\\_frozen.json} before any detector output existed).",
          "tab:config", "lp{0.72\\linewidth}", "Component & Setting", rows, size="\\scriptsize")


def main():
    t_leakage(); t_truncation(); t_partition(); t_partition("R7-ledger-fcos-T320", "TABLE_partition_fcos.tex", "FCOS")
    t_methods(); t_methods("R5R9-methods-fcos", "TABLE_methods_fcos.tex", "FCOS"); t_ccpr(); t_sensitivity(); t_yst(); t_config()


if __name__ == "__main__":
    main()
