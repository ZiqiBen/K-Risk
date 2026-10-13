"""Generate the 372 x 2 LLM annotations with the system prompt of release/system_prompt.py.

Same inputs as the released annotations (the decision-frame description of each
event, i.e. the last described frame), same decoding (temperature 0, one generation,
at most 2,500 output tokens), dated model snapshots. Responses are written to a
staging folder; nothing in the release is touched by this script.

Access: the OpenAI API when OPENAI_API_KEY is set; otherwise OpenRouter (OpenAI-compatible
endpoint, provider pinned to OpenAI) with the key in OPENROUTER_API_KEY.

Run: python 13_regenerate_llm.py [--models openai/gpt-4o-2024-08-06 openai/gpt-4.1] [--workers 6]
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "release"))
from system_prompt import SYSTEM_MESSAGE as SYSTEM  # noqa: E402

ROOT = os.path.join(os.environ.get("KR_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "K-Risk_data")), "llm_analysis")
DESC = os.path.join(ROOT, "event_discriptions")
STAGE = os.environ.get("KR_STAGE", os.path.join(os.path.dirname(__file__), "llm_v3"))
FOLDER = {"gpt-4o": "analysis_gpt4o", "gpt-4.1": "analysis_gpt4.1"}


def api_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        sys.exit("Set OPENAI_API_KEY or OPENROUTER_API_KEY")
    return k


def last_frame_text(content):
    """Identical to llm_test.ipynb::extract_last_frame_text."""
    frames = list(re.finditer(r"Frame (\d+):", content))
    return content[frames[-1].start():].strip()


def family(model):
    return "gpt-4.1" if "gpt-4.1" in model else "gpt-4o"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["openai/gpt-4o-2024-08-06", "openai/gpt-4.1"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="only the first N events (dry run)")
    ap.add_argument("--icl", help="text file appended to the system prompt (in-context failure cases)")
    ap.add_argument("--skip", help="csv with an 'event' column; these events are not run")
    a = ap.parse_args()
    system = SYSTEM
    if a.icl:
        system = (SYSTEM.rstrip() + "\n\n" + open(a.icl, encoding="utf-8").read().strip()
                  + "\n\nNow decide for the scenario below. Your response must end with an action ID in the "
                  "format: Final Decision: ####<ID>")
    from openai import OpenAI
    direct = bool(os.environ.get("OPENAI_API_KEY"))
    client = OpenAI() if direct else OpenAI(api_key=api_key(), base_url="https://openrouter.ai/api/v1")
    files = sorted(f for f in os.listdir(DESC) if f.endswith(".txt"))
    if a.skip:
        import csv
        skip = {"extreme_desc_" + r["event"] + ".txt" for r in csv.DictReader(open(a.skip, encoding="utf-8"))}
        files = [f for f in files if f not in skip]
    if a.limit:
        files = files[: a.limit]
    os.makedirs(STAGE, exist_ok=True)
    log = open(os.path.join(STAGE, "generation_log.jsonl"), "a")

    def run(job):
        model, fn = job
        out_dir = os.path.join(STAGE, FOLDER[family(model)])
        out = os.path.join(out_dir, "response_" + fn)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return
        user = last_frame_text(open(os.path.join(DESC, fn), encoding="utf-8").read())
        for attempt in range(5):
            try:
                route = (dict(model=model.removeprefix("openai/")) if direct else
                         dict(model=model, extra_body={"provider": {"order": ["OpenAI"], "allow_fallbacks": False}}))
                r = client.chat.completions.create(
                    temperature=0.0, max_tokens=2500, n=1,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **route)
                text = r.choices[0].message.content
                with open(out, "w", encoding="utf-8") as f:
                    f.write(text)
                extra = getattr(r, "model_extra", None) or {}
                log.write(json.dumps(dict(file=fn, requested=model, returned=r.model, provider=extra.get("provider"),
                                          fingerprint=getattr(r, "system_fingerprint", None),
                                          prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens,
                                          finish=r.choices[0].finish_reason,
                                          time=dt.datetime.utcnow().isoformat())) + "\n")
                log.flush()
                return
            except Exception as exc:  # noqa: BLE001  (API errors only; content is never resampled)
                wait = min(60, 2 ** (attempt + 1))
                print(f"retry {attempt + 1} {model} {fn}: {exc!r}"[:300], flush=True)
                time.sleep(wait)
        print("FAILED", model, fn, flush=True)

    for m in a.models:
        os.makedirs(os.path.join(STAGE, FOLDER[family(m)]), exist_ok=True)
    jobs = [(m, f) for m in a.models for f in files]
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))
    print("done", len(jobs))


if __name__ == "__main__":
    main()
