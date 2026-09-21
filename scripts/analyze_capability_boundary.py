"""
analyze_capability_boundary.py
================================
Step 5 + 6: Analyze probe evaluation results and determine the capability
boundary of the 124M local model. Also checks whether the existing API
judge can identify reliable cases and considers routing implications.

Reads:
  results/final/capability_probe_evaluation.json
  results/final/capability_by_category.json

Produces:
  results/final/capability_boundary_report.md
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_PATH   = PROJECT_ROOT / "results" / "final" / "capability_probe_evaluation.json"
CAT_PATH    = PROJECT_ROOT / "results" / "final" / "capability_by_category.json"
REPORT_PATH = PROJECT_ROOT / "results" / "final" / "capability_boundary_report.md"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_log_prob_correlation(evaluated):
    """
    Check if avg_log_prob correlates with quality.
    Returns a dict with mean log-prob per quality label and overall correlation signal.
    """
    quality_lp = defaultdict(list)
    for r in evaluated:
        lp = r.get("avg_log_prob")
        if lp is not None:
            quality_lp[r["quality"]].append(lp)

    result = {}
    for q, lps in quality_lp.items():
        result[q] = {
            "n": len(lps),
            "mean_log_prob": round(sum(lps)/len(lps), 4) if lps else None,
            "min": round(min(lps), 4) if lps else None,
            "max": round(max(lps), 4) if lps else None,
        }
    return result


def get_examples(evaluated, quality, n=2):
    """Get up to n examples of given quality."""
    examples = [r for r in evaluated if r["quality"] == quality]
    return examples[:n]


def get_failure_mode_distribution(evaluated):
    modes = defaultdict(int)
    for r in evaluated:
        if r["quality"] in ("BAD", "UNKNOWN"):
            modes[r.get("failure_mode", "other")] += 1
    return dict(sorted(modes.items(), key=lambda x: -x[1]))


def determine_tier1_safe_classes(cat_summary, threshold_strict=0.5, threshold_broad=0.65):
    """
    Returns categories where the model is arguably reliable.
    Uses strict (GOOD-only) and broad (GOOD+PARTIAL) thresholds.
    """
    safe = []
    marginal = []
    unsafe = []
    for cat, d in cat_summary.items():
        strict = d["success_rate_strict"]
        broad  = d["success_rate_broad"]
        if strict >= threshold_strict and broad >= threshold_broad:
            safe.append(cat)
        elif broad >= 0.40:
            marginal.append(cat)
        else:
            unsafe.append(cat)
    return safe, marginal, unsafe


def format_example(r, label):
    return (
        f"**{label}** (`{r['question_id']}`)\n"
        f"- **Q:** {r['question']}\n"
        f"- **GT:** {r['ground_truth'][:150]}\n"
        f"- **Generated:** {r['generated_answer'][:200]}\n"
        f"- **Quality:** {r['quality']} (score={r['score']})\n"
        f"- **Reason:** {r['reason']}\n"
        f"- **Failure mode:** {r['failure_mode']}\n"
        f"- **Avg log-prob:** {r.get('avg_log_prob')}\n"
    )


def write_report(cat_summary, evaluated):
    total = len(evaluated)
    total_good = sum(1 for r in evaluated if r["quality"] == "GOOD")
    total_partial = sum(1 for r in evaluated if r["quality"] == "PARTIALLY_GOOD")
    total_bad = sum(1 for r in evaluated if r["quality"] == "BAD")
    total_unk = sum(1 for r in evaluated if r["quality"] == "UNKNOWN")
    overall_strict = total_good / total if total else 0
    overall_broad  = (total_good + total_partial) / total if total else 0

    lp_corr = analyze_log_prob_correlation(evaluated)
    failure_modes = get_failure_mode_distribution(evaluated)
    safe, marginal, unsafe = determine_tier1_safe_classes(cat_summary)

    lines = [
        "# EdgeCascade Capability Boundary Report",
        "",
        "> **Analysis Date:** Auto-generated after running capability probe.",
        "> **Model:** Custom GPT-2 124M + Q/V LoRA (rank=8, alpha=16) + SFT adapter (30 Q&A pairs, 8 epochs)",
        "> **Probe size:** 40 questions across 7 categories (ISRO + DPDPA domains)",
        "",
        "---",
        "",
        "## 1. Overall Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total probe questions | {total} |",
        f"| GOOD | {total_good} ({total_good/total*100:.1f}%) |",
        f"| PARTIALLY_GOOD | {total_partial} ({total_partial/total*100:.1f}%) |",
        f"| BAD | {total_bad} ({total_bad/total*100:.1f}%) |",
        f"| UNKNOWN | {total_unk} ({total_unk/total*100:.1f}%) |",
        f"| **Strict success rate (GOOD only)** | **{overall_strict*100:.1f}%** |",
        f"| **Broad success rate (GOOD + PARTIAL)** | **{overall_broad*100:.1f}%** |",
        "",
        "---",
        "",
        "## 2. Results by Category",
        "",
        "| Category | N | GOOD | PARTIAL | BAD | UNK | Strict% | Broad% | Avg logP |",
        "|----------|---|------|---------|-----|-----|---------|--------|----------|",
    ]

    for cat, d in sorted(cat_summary.items()):
        lp_str = f"{d['avg_log_prob']:.3f}" if d.get('avg_log_prob') is not None else "N/A"
        lines.append(
            f"| {cat} | {d['n_questions']} | {d['GOOD']} | {d['PARTIALLY_GOOD']} | "
            f"{d['BAD']} | {d['UNKNOWN']} | {d['success_rate_strict']*100:.0f}% | "
            f"{d['success_rate_broad']*100:.0f}% | {lp_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Capability Boundary Analysis",
        "",
        "### 3.1 What can the 124M model reliably do?",
        "",
    ]

    if safe:
        lines.append(
            f"Categories meeting Tier-1-safe thresholds "
            f"(strict ≥ 50%, broad ≥ 65%): **{', '.join(safe)}**"
        )
        lines.append("")
        lines.append("These categories are candidates for Tier-1 acceptance, provided the API judge also passes the answer.")
        for cat in safe:
            lines.append(f"\n#### {cat}")
            exs = get_examples([r for r in evaluated if r["category"] == cat and r["quality"] == "GOOD"], None, 1)
            if exs:
                lines.append(format_example(exs[0], "Successful example"))
    else:
        lines.append(
            "**No categories reached the defined Tier-1-safe thresholds (strict ≥ 50%, broad ≥ 65%).**"
        )
        lines.append("")
        lines.append(
            "This is a scientifically honest negative result. The 124M SFT model does not have a "
            "meaningfully reliable capability region at current training scale."
        )

    if marginal:
        lines += [
            "",
            "### 3.2 Marginal categories (broad 40-65%)",
            "",
            f"Categories where the model sometimes succeeds but is not reliable: **{', '.join(marginal)}**",
            "",
            "These categories have success but too much variance to safely use as Tier-1 without additional signals.",
        ]

    if unsafe:
        lines += [
            "",
            "### 3.3 Categories that are unsafe for Tier 1",
            "",
            f"Categories where the model consistently fails (broad < 40%): **{', '.join(unsafe)}**",
            "",
        ]

    # Failure mode distribution
    lines += [
        "",
        "### 3.4 Failure mode distribution (BAD + UNKNOWN answers)",
        "",
        "| Failure Mode | Count |",
        "|--------------|-------|",
    ]
    for mode, count in failure_modes.items():
        lines.append(f"| {mode} | {count} |")

    # Failure examples
    bad_examples = get_examples([r for r in evaluated if r["quality"] == "BAD"], None, 2)
    if bad_examples:
        lines += [
            "",
            "### 3.5 Representative failure examples",
            "",
        ]
        for ex in bad_examples:
            lines.append(format_example(ex, "Failure"))
            lines.append("")

    # Log-probability analysis
    lines += [
        "",
        "---",
        "",
        "## 4. Log-Probability as a Quality Signal",
        "",
        "| Quality Label | N | Mean Avg-LogP | Min | Max |",
        "|---------------|---|---------------|-----|-----|",
    ]
    for qual in ["GOOD", "PARTIALLY_GOOD", "BAD", "UNKNOWN"]:
        d = lp_corr.get(qual, {})
        n = d.get("n", 0)
        mean = d.get("mean_log_prob", "N/A")
        mn = d.get("min", "N/A")
        mx = d.get("max", "N/A")
        lines.append(f"| {qual} | {n} | {mean} | {mn} | {mx} |")

    # Determine if log-prob is useful
    good_lp = lp_corr.get("GOOD", {}).get("mean_log_prob")
    bad_lp  = lp_corr.get("BAD",  {}).get("mean_log_prob")
    if good_lp is not None and bad_lp is not None:
        diff = good_lp - bad_lp
        if diff > 0.2:
            lp_useful = True
            lp_verdict = (
                f"Log-probability shows a positive correlation with quality: "
                f"GOOD answers average {good_lp:.3f} vs BAD answers at {bad_lp:.3f} "
                f"(difference: {diff:.3f}). This signal may be useful for routing, "
                f"but cannot be used alone without the API judge."
            )
        elif diff < -0.05:
            lp_useful = False
            lp_verdict = (
                f"Log-probability is **inversely** correlated or uncorrelated with quality: "
                f"GOOD={good_lp:.3f}, BAD={bad_lp:.3f}. The model produces bad answers with high confidence. "
                f"Log-probability is NOT a reliable signal for routing decisions."
            )
        else:
            lp_useful = False
            lp_verdict = (
                f"Log-probability shows minimal separation between quality levels: "
                f"GOOD={good_lp:.3f}, BAD={bad_lp:.3f}. Not sufficient as a standalone routing signal."
            )
    else:
        lp_useful = False
        lp_verdict = "Insufficient data to determine log-probability utility."

    lines += ["", lp_verdict, ""]

    # Confidently wrong examples (high log-prob but BAD)
    confident_wrong = [
        r for r in evaluated
        if r["quality"] == "BAD" and r.get("avg_log_prob") is not None and r["avg_log_prob"] > -1.5
    ]
    if confident_wrong:
        lines += [
            "### 4.1 Confidently wrong examples (high log-prob + BAD quality)",
            "",
        ]
        for ex in confident_wrong[:2]:
            lines.append(format_example(ex, "Confidently wrong"))
            lines.append("")

    # Routing section
    lines += [
        "",
        "---",
        "",
        "## 5. Routing Implications",
        "",
    ]

    if safe:
        lines += [
            f"A meaningful reliable capability region was identified in categories: **{', '.join(safe)}**",
            "",
            "### Recommended routing change",
            "",
            "The existing API judge already evaluates Tier-1 answers. Since the API judge passes answers ",
            "only when they are factually coherent and relevant, no routing threshold changes are needed. ",
            "",
            "The recommended approach is to:",
            "1. Keep the current API judge as the Tier-1 gate (no threshold lowering).",
            "2. Accept that on the probe categories where the model is reliable, the judge will naturally pass.",
            "3. Do NOT hard-code category-based routing; let the judge decide organically.",
            "",
            "> **Verdict: KEEP 124M as Tier 1 with capability-aware routing** — but only for the categories ",
            "> identified above, and only when the API judge independently passes the answer.",
        ]
    else:
        lines += [
            "No category achieved a success rate warranting a reliable Tier-1 routing policy.",
            "",
            "The model's failure modes (hallucination, repetition, instruction-following failures) are too ",
            "pervasive and unpredictable. Even in categories where some answers are correct, the judge would ",
            "need to catch all failures — and the failure rate is high enough that the benefit of Tier-1 usage ",
            "is negligible.",
            "",
            "> **Verdict: 124M HAS NO MEANINGFUL SAFE CAPABILITY REGION.** ",
            "> Keep it as the from-scratch model component demonstrating the architecture, ",
            "> but do not rely on it for practical Tier-1 routing.",
        ]

    lines += [
        "",
        "---",
        "",
        "## 6. Final Recommendation",
        "",
    ]

    if safe:
        lines += [
            "**OPTION A: KEEP 124M AS TIER 1 AND USE CAPABILITY-AWARE ROUTING**",
            "",
            f"Evidence: Categories {safe} show ≥50% strict success rate in offline evaluation.",
            "The API judge already acts as the gate, so no threshold changes are needed.",
            "Tier-1 usage will remain low but honest — only when the judge independently agrees.",
        ]
    else:
        lines += [
            "**OPTION B: 124M HAS NO MEANINGFUL SAFE CAPABILITY REGION**",
            "",
            "Keep it as the from-scratch architecture demonstration. ",
            "Do not force Tier-1 routing. The cascade correctly escalates almost everything,",
            "which is the scientifically honest result given the model's scale and training.",
        ]

    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {REPORT_PATH}")


def main():
    print("Analyzing capability probe results ...")

    if not EVAL_PATH.exists():
        print(f"ERROR: {EVAL_PATH} not found. Run evaluate_capability_probe.py first.")
        sys.exit(1)

    evaluated = load_json(EVAL_PATH)
    cat_summary = load_json(CAT_PATH)

    write_report(cat_summary, evaluated)
    print("Done.")


if __name__ == "__main__":
    main()
