"""BEAM judge runner.

For each (question, llm_response) pair our bench produced, call the
configured LLM judge with **the exact upstream BEAM judge prompt**, then
parse the per-nugget score and aggregate per question / per category.

The judge prompt itself is imported VERBATIM from the upstream submodule:

    from prompts import unified_llm_judge_base_prompt

No prompt text lives in our codebase. Anyone wanting to verify can diff
``benchmarks/beam/upstream/src/prompts.py`` against the same SHA on
``github.com/mohammadtavakoli78/BEAM`` — same prompt, same logic.

What we DO NOT do here (intentionally, to keep deps light):
* invoke upstream's ``compute_metrics.py``, which depends on spacy,
  sentence-transformers, scipy, nltk, rouge_score etc. (~50 heavy deps).
  BEAM's BLEU/ROUGE/cosine-similarity helpers are useful for some
  question types but the headline score is the LLM judge's nugget
  average, and that's what we replicate exactly here.

If you want to invoke upstream's evaluator directly (full standards
compliance with all metrics), install ``benchmarks/beam/upstream/requirements.txt``
into a separate venv and run::

    python -m src.evaluation.run_evaluation \
        --probing_questions <path-to-our-probing_questions.json> \
        --answers_directory  <path-to-our-answers.json> \
        --output             <output_path>

Our ``answers.json`` files are already in the exact format their evaluator
expects.

CLI::

    python -m benchmarks.beam.judge --run-dir benchmarks/beam/runs/<run_id>
    python -m benchmarks.beam.judge --run-dir <run_id> --only-conv 1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from benchmarks.beam.config import (
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_MODEL,
    REPO_ROOT,
)

# ---- Import the upstream judge prompt VERBATIM -------------------------------

_UPSTREAM_SRC = REPO_ROOT / "benchmarks" / "beam" / "upstream" / "src"
sys.path.insert(0, str(_UPSTREAM_SRC))
try:
    from prompts import unified_llm_judge_base_prompt  # type: ignore  # noqa: E402
finally:
    # Don't pollute sys.path for any sibling imports
    try:
        sys.path.remove(str(_UPSTREAM_SRC))
    except ValueError:
        pass


# Sanity check at import time: the upstream prompt must contain the three
# placeholders we replace. If upstream changes them, we want to fail loudly,
# not silently send a wrong prompt to the judge.
for _ph in ("<question>", "<rubric_item>", "<llm_response>"):
    if _ph not in unified_llm_judge_base_prompt:
        raise RuntimeError(
            f"upstream judge prompt missing expected placeholder {_ph!r}; "
            "submodule may have drifted from the SHA this code was written against."
        )


# ---- LLM call (OpenAI-compatible chat completions) --------------------------

def _resolve_qwen_model() -> str:
    """If QWEN_MODEL is unset, ask vLLM what it's serving and use that."""
    if QWEN_MODEL:
        return QWEN_MODEL
    try:
        r = requests.get(f"{QWEN_BASE_URL}/models", timeout=10,
                         headers={"Authorization": f"Bearer {QWEN_API_KEY}"})
        r.raise_for_status()
        data = r.json().get("data", [])
        if data:
            return data[0]["id"]
    except Exception:
        pass
    return "qwen"  # final fallback — vLLM usually accepts any string


def _call_judge(prompt_text: str, model_id: str, timeout: float = 180) -> str:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {QWEN_API_KEY}",
    }
    r = requests.post(
        f"{QWEN_BASE_URL}/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    r.raise_for_status()
    body = r.json()
    return body["choices"][0]["message"]["content"]


# A self-hosted judge that reasons at length can miss a deadline or drop a
# connection under load. Without retries such a nugget scored 0.0, silently,
# exactly like a real "not met" - so the arm that hit more timeouts looked
# worse. Transport failures are now retried with backoff, and a nugget that
# still fails is COUNTED as a judge error (kept separate from parse errors)
# so a run with errors is visible and can be re-judged.
_RETRY_BACKOFF_S = (10, 30, 60)


def _call_judge_with_retries(prompt_text: str, model_id: str, timeout: float,
                             retries: int) -> str:
    for attempt in range(retries + 1):
        try:
            return _call_judge(prompt_text, model_id, timeout=timeout)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(_RETRY_BACKOFF_S[min(attempt, len(_RETRY_BACKOFF_S) - 1)])
    raise AssertionError("unreachable")


# ---- Verdict parsing --------------------------------------------------------

_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9.]+)')


def _parse_judge_response(response_text: str) -> dict:
    """Return ``{"score": float in {0.0, 0.5, 1.0}, "raw": str, ...}``.

    Upstream uses ``json_repair.repair_json`` to be robust to malformed JSON.
    We try the obvious json.loads first, then a regex fallback. If neither
    yields a parseable score, we record score=0.0 + the raw text so the
    reviewer can spot-check. Upstream also clamps to {0, 0.5, 1.0}.
    """
    text = (response_text or "").strip()
    # Strip code-fence wrappers if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        m = _SCORE_RE.search(text)
        if m:
            parsed = {"score": float(m.group(1))}

    if not isinstance(parsed, dict) or "score" not in parsed:
        return {"score": 0.0, "parse_error": True, "raw": text[:500]}

    try:
        score = float(parsed["score"])
    except (TypeError, ValueError):
        return {"score": 0.0, "parse_error": True, "raw": text[:500]}

    # Clamp to allowed values per BEAM rubric scheme (0 / 0.5 / 1.0)
    if score <= 0.25:
        score = 0.0
    elif score <= 0.75:
        score = 0.5
    else:
        score = 1.0

    parsed["score"] = score
    parsed["raw"] = text[:500]
    return parsed


# ---- Per-conversation scoring -----------------------------------------------

def _judge_one_question(
    question: str,
    llm_response: str,
    rubric: list[str],
    model_id: str,
    timeout: float,
    retries: int = 0,
) -> dict:
    nugget_results = []
    cumulative = 0.0
    parse_errors = 0
    judge_errors = 0
    for nugget in rubric:
        prompt_text = (
            unified_llm_judge_base_prompt
            .replace("<question>", question)
            .replace("<rubric_item>", nugget)
            .replace("<llm_response>", llm_response)
        )
        try:
            raw = _call_judge_with_retries(prompt_text, model_id, timeout, retries)
            verdict = _parse_judge_response(raw)
        except Exception as e:
            verdict = {"score": 0.0, "error": f"{type(e).__name__}: {e}"}
            judge_errors += 1
        if verdict.get("parse_error"):
            parse_errors += 1
        nugget_results.append({"nugget": nugget, "verdict": verdict})
        cumulative += float(verdict.get("score", 0.0))
    avg = cumulative / len(rubric) if rubric else 0.0
    return {
        "nugget_count": len(rubric),
        "nugget_average_score": round(avg, 4),
        "parse_errors": parse_errors,
        "judge_errors": judge_errors,
        "nuggets": nugget_results,
    }


def judge_conversation(conv_dir: Path, model_id: str, timeout: float = 180,
                       workers: int = 1, retries: int = 0) -> dict:
    """Score every (answer, rubric) pair in one conv subdirectory.

    Reads ``answers.json`` + ``probing_questions.json``. Writes
    ``verdicts.json`` (per-question detail) + ``category_scores.json``
    (per-category aggregates) into the same conv subdir. Returns the
    aggregate dict.
    """
    answers = json.loads((conv_dir / "answers.json").read_text(encoding="utf-8"))
    probing = json.loads((conv_dir / "probing_questions.json").read_text(encoding="utf-8"))

    started = time.monotonic()
    by_category: dict[str, list[dict]] = {}
    category_scores: dict[str, dict] = {}

    def _judge(category: str, i: int, n: int, question: str, response: str,
               rubric: list[str]) -> dict:
        qstart = time.monotonic()
        result = _judge_one_question(question, response, rubric, model_id, timeout, retries)
        result["question"] = question
        result["llm_response_preview"] = (response or "")[:300]
        result["elapsed_s"] = round(time.monotonic() - qstart, 2)
        print(
            f"  [{conv_dir.name} {category} {i+1}/{n}] "
            f"score={result['nugget_average_score']:.2f} "
            f"nuggets={result['nugget_count']} "
            f"parse_err={result['parse_errors']} "
            f"judge_err={result['judge_errors']} "
            f"({result['elapsed_s']}s)",
            flush=True,
        )
        return result

    # Questions are independent, so `workers` > 1 judges them in parallel;
    # each question's nuggets stay sequential and results keep their order.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending: dict[str, list] = {}
        for category, questions in answers.items():
            rubrics = [item.get("rubric", []) for item in probing.get(category, [])]
            slots: list = []
            for i, ans_item in enumerate(questions):
                question = ans_item["question"]
                response = ans_item["llm_response"]
                rubric = rubrics[i] if i < len(rubrics) else []
                if not rubric:
                    slots.append({
                        "question": question,
                        "skipped": True,
                        "reason": "no rubric for this index",
                    })
                    continue
                slots.append(pool.submit(_judge, category, i, len(questions),
                                         question, response, rubric))
            pending[category] = slots
        for category, slots in pending.items():
            by_category[category] = [s.result() if hasattr(s, "result") else s for s in slots]

    for category, per_q in by_category.items():
        non_skipped = [q for q in per_q if "skipped" not in q]
        cat_avg = (
            sum(q["nugget_average_score"] for q in non_skipped) / len(non_skipped)
            if non_skipped else 0.0
        )
        category_scores[category] = {
            "question_count": len(non_skipped),
            "average_score": round(cat_avg, 4),
        }

    overall_questions = [
        q for cat in by_category.values() for q in cat if "skipped" not in q
    ]
    overall_avg = (
        sum(q["nugget_average_score"] for q in overall_questions) / len(overall_questions)
        if overall_questions else 0.0
    )

    aggregate = {
        "overall_average_score": round(overall_avg, 4),
        "total_questions_judged": len(overall_questions),
        "judge_errors": sum(q.get("judge_errors", 0) for q in overall_questions),
        "categories": category_scores,
        "model_id": model_id,
        "judge_endpoint": QWEN_BASE_URL,
        "wall_clock_s": round(time.monotonic() - started, 1),
    }

    (conv_dir / "verdicts.json").write_text(
        json.dumps(by_category, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (conv_dir / "category_scores.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate


# ---- CLI --------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="path to a run dir under benchmarks/beam/runs/")
    p.add_argument("--only-conv", default=None,
                   help="comma-separated conversation_ids to judge (default: all)")
    p.add_argument("--timeout", type=float, default=180,
                   help="per-judge-call HTTP timeout")
    p.add_argument("--model", default=None,
                   help="override judge model id (otherwise auto-resolved from vLLM /v1/models)")
    p.add_argument("--workers", type=int, default=1,
                   help="questions judged in parallel (default 1 = sequential)")
    p.add_argument("--retries", type=int, default=2,
                   help="retries per nugget on timeouts / dropped connections before it is "
                        "counted as a judge error")
    p.add_argument("--require-endpoint", default=None,
                   help="refuse to start unless the judge endpoint starts with this prefix, so "
                        "answers can never be sent to the hosted default by mistake")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"run dir not found: {run_dir}")
        return 1

    if args.require_endpoint and not QWEN_BASE_URL.startswith(args.require_endpoint):
        print(f"REFUSING: judge endpoint is not {args.require_endpoint!r} "
              f"(set QWEN_BASE_URL); nothing was sent")
        return 2
    model_id = args.model or _resolve_qwen_model()
    print(f"judge: {QWEN_BASE_URL}  model={model_id}  workers={args.workers}")

    only_ids = None
    if args.only_conv:
        only_ids = {s.strip() for s in args.only_conv.split(",")}

    conv_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("conv_"))
    if only_ids:
        conv_dirs = [p for p in conv_dirs if p.name.split("_")[1].lstrip("0") in only_ids]

    if not conv_dirs:
        print(f"no conversation subdirs found under {run_dir}")
        return 1

    aggregates: list[dict] = []
    for conv_dir in conv_dirs:
        print(f"\n--- {conv_dir.name} ---")
        try:
            agg = judge_conversation(conv_dir, model_id, timeout=args.timeout,
                                     workers=args.workers, retries=args.retries)
            aggregates.append({"conv": conv_dir.name, **agg})
        except Exception as e:
            print(f"!!! failed: {e}")
            aggregates.append({"conv": conv_dir.name, "fatal_error": str(e)})

    summary = _aggregate_run(aggregates)
    (run_dir / "judge_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {run_dir / 'judge_summary.json'}")
    print(f"overall: {summary['overall_average_score']:.4f} "
          f"({summary['total_questions_judged']} questions)")
    return 0


def _aggregate_run(per_conv: list[dict]) -> dict:
    """Aggregate per-conv aggregates into a run-level summary."""
    cat_sums: dict[str, dict] = {}
    total_qs = 0
    overall_num = 0.0

    for agg in per_conv:
        if "fatal_error" in agg:
            continue
        total_qs += agg.get("total_questions_judged", 0)
        overall_num += agg.get("overall_average_score", 0.0) * agg.get("total_questions_judged", 0)
        for cat, cat_agg in agg.get("categories", {}).items():
            entry = cat_sums.setdefault(cat, {"question_count": 0, "weighted_score": 0.0})
            qc = cat_agg.get("question_count", 0)
            entry["question_count"] += qc
            entry["weighted_score"] += cat_agg.get("average_score", 0.0) * qc

    cat_scores = {
        cat: {
            "question_count": entry["question_count"],
            "average_score": (
                round(entry["weighted_score"] / entry["question_count"], 4)
                if entry["question_count"] else 0.0
            ),
        }
        for cat, entry in cat_sums.items()
    }

    return {
        "overall_average_score": round(overall_num / total_qs, 4) if total_qs else 0.0,
        "total_questions_judged": total_qs,
        "conversations_judged": sum(1 for a in per_conv if "fatal_error" not in a),
        "conversations_attempted": len(per_conv),
        "categories": cat_scores,
        "per_conversation": per_conv,
    }


if __name__ == "__main__":
    raise SystemExit(main())
