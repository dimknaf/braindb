"""Phase C only: answer probing questions for one already-ingested conversation.

Reuses ``answer_one_question`` and ``load_probing_questions`` from the main
bench. Skips Phase A (DB create + api restart + ingest) and Phase B (warmup
wait) entirely; the conv DB must already be populated by a prior run.

Used to score conversations whose DB has real facts but whose Phase C
failed in the original run (e.g. vLLM was down during Phase C only).

The new artefacts land in the SAME shape ``judge.py``/``score.py`` already
expect (answers.json, probing_questions.json, raw_payloads.json,
conversation_meta.json), under ``--output-dir``.

CLI:

    python -m benchmarks.beam.phase_c_only \\
        --split 100K --conv-id 1 \\
        --api-base http://api_bench_conv1:8001 \\
        --output-dir /app/benchmarks/beam/runs/<run_id>/conv_001

    # questions from a file instead of the dataset; record the DB really used
    python -m benchmarks.beam.phase_c_only \\
        --probing-json <path>/probing_questions.json --db-label <db name> \\
        --api-base <api> --output-dir <run_dir>/conv_015
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

from benchmarks.beam.adapter import iter_conversations, load_probing_questions
from benchmarks.beam.bench import QUESTION_TIMEOUT_SECONDS, answer_one_question


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--split", default="100K", choices=["100K", "500K", "1M"])
    p.add_argument("--conv-id", default=None,
                   help="BEAM conversation_id field (e.g. '1'); optional with --probing-json")
    p.add_argument("--probing-json", default=None,
                   help="read the probing questions from this file (the probing_questions.json "
                        "shape) instead of the BEAM dataset: archived questions, or a question "
                        "set written for another database")
    p.add_argument("--db-label", default=None,
                   help="the database the API at --api-base really uses, recorded in "
                        "conversation_meta.json (default: braindb_conv_<id>)")
    p.add_argument("--api-base", required=True,
                   help="braindb API base, e.g. http://api_bench_conv1:8001")
    p.add_argument("--output-dir", required=True,
                   help="conv_NNN directory to write artifacts into")
    p.add_argument("--question-timeout", type=float,
                   default=QUESTION_TIMEOUT_SECONDS)
    p.add_argument("--errors-only", action="store_true",
                   help="Read existing answers.json in --output-dir and re-ask "
                        "only questions whose llm_response starts with '[ERROR:'. "
                        "Successful answers are kept verbatim.")
    return p.parse_args()


def _load_existing(out: Path) -> tuple[dict, dict]:
    """Load existing answers.json + raw_payloads.json from output dir, if any."""
    answers = {}
    payloads = {}
    ap = out / "answers.json"
    rp = out / "raw_payloads.json"
    if ap.exists():
        answers = json.loads(ap.read_text(encoding="utf-8"))
    if rp.exists():
        payloads = json.loads(rp.read_text(encoding="utf-8"))
    return answers, payloads


def main() -> int:
    args = _parse_args()
    out = Path(args.output_dir)

    if args.probing_json:
        src = Path(args.probing_json)
        probing = json.loads(src.read_text(encoding="utf-8"))
        conv_id, split, slug, seed_category = args.conv_id, None, src.stem, None
    else:
        if not args.conv_id:
            raise SystemExit("--conv-id is required unless --probing-json is given")
        target = None
        for conv in iter_conversations(args.split):
            if str(conv.conversation_id) == str(args.conv_id):
                target = conv
                break
        if target is None:
            raise SystemExit(
                f"conversation_id={args.conv_id} not found in split {args.split}"
            )
        probing = load_probing_questions(target)
        conv_id, split, slug = target.conversation_id, target.split, target.slug
        seed_category = (target.raw.get("conversation_seed") or {}).get("category")
    total = sum(len(v) for v in probing.values())

    existing_answers: dict[str, list[dict]] = {}
    existing_payloads: dict[str, list[dict]] = {}
    if args.errors_only:
        existing_answers, existing_payloads = _load_existing(out)
        if not existing_answers:
            raise SystemExit(
                f"--errors-only set but no answers.json found at {out}"
            )

    mode = "errors-only" if args.errors_only else "full"
    print(
        f"[C-only/{mode}] conv {slug}: {total} questions "
        f"across {len(probing)} categories, api={args.api_base}",
        flush=True,
    )

    answers_by_category: dict[str, list[dict]] = {}
    raw_payloads_by_category: dict[str, list[dict]] = {}
    question_errors = 0
    questions_kept = 0
    started = time.monotonic()

    for category, questions in probing.items():
        category_answers: list[dict] = []
        category_payloads: list[dict] = []
        existing_cat = existing_answers.get(category, [])
        existing_cat_payloads = existing_payloads.get(category, [])
        for i, q in enumerate(questions, start=1):
            question_text = q["question"]
            idx = i - 1
            # If --errors-only and existing answer is good, keep it verbatim
            if (args.errors_only
                    and idx < len(existing_cat)
                    and not str(existing_cat[idx].get("llm_response", "")).startswith("[ERROR:")):
                category_answers.append(existing_cat[idx])
                if idx < len(existing_cat_payloads):
                    category_payloads.append(existing_cat_payloads[idx])
                else:
                    category_payloads.append({
                        "question": question_text,
                        "payload": {"_kept": True}
                    })
                questions_kept += 1
                print(
                    f"  [{category} {i}/{len(questions)}] KEPT existing OK answer",
                    flush=True,
                )
                continue
            try:
                ans, payload = answer_one_question(
                    question_text,
                    base=args.api_base,
                    timeout=args.question_timeout,
                )
                print(
                    f"  [{category} {i}/{len(questions)}] OK "
                    f"({payload.get('_elapsed_seconds', '?')}s, {len(ans)} chars)",
                    flush=True,
                )
            except Exception as e:
                ans = f"[ERROR: {type(e).__name__}: {e}]"
                payload = {"_error": str(e)}
                question_errors += 1
                print(
                    f"  [{category} {i}/{len(questions)}] ERROR: {e}",
                    flush=True,
                )
            category_answers.append(
                {"question": question_text, "llm_response": ans}
            )
            category_payloads.append(
                {"question": question_text, "payload": payload}
            )
        answers_by_category[category] = category_answers
        raw_payloads_by_category[category] = category_payloads

    out.mkdir(parents=True, exist_ok=True)
    out.joinpath("answers.json").write_text(
        json.dumps(answers_by_category, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    out.joinpath("probing_questions.json").write_text(
        json.dumps(probing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    out.joinpath("raw_payloads.json").write_text(
        json.dumps(raw_payloads_by_category, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    meta = {
        "conversation_id": conv_id,
        "split": split,
        "slug": slug,
        "category": seed_category,
        "ingested_md_size_kb": None,
        "question_errors": question_errors,
        "questions_kept": questions_kept,
        "wall_clock_s": round(time.monotonic() - started, 1),
        "started_at": _now_iso(),
        "bench_db_name": args.db_label or (f"braindb_conv_{int(conv_id):03d}" if conv_id else None),
        "phase_c_only": True,
        "errors_only_mode": args.errors_only,
        "api_base": args.api_base,
    }
    out.joinpath("conversation_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(
        f"[done] {slug}, kept={questions_kept}, "
        f"errors={question_errors}/{total - questions_kept} new, "
        f"wall_clock_s={meta['wall_clock_s']:.0f}s, dir={out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
