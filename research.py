"""research.py - Main research runner.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import tempfile
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^\w]+", "-", str(topic or "").strip().lower(), flags=re.UNICODE).strip("-")
    slug = slug[:60].rstrip("-")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Create a deep research survey about this topic: {str(topic).strip()}\n\n"
        "Follow your system workflow exactly. Delegate at least three independent research questions, inspect the notes, "
        "create the source registry and report in the specified sandbox paths, then run the finalizer, validator, and "
        "citation spot-check before finishing."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    calls = Counter()
    input_tokens = 0
    output_tokens = 0
    for message in messages or []:
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls") or []
            usage = message.get("usage_metadata") or {}
        else:
            tool_calls = getattr(message, "tool_calls", None) or []
            usage = getattr(message, "usage_metadata", None) or {}
        for call in tool_calls:
            if isinstance(call, dict):
                name = call.get("name") or (call.get("function") or {}).get("name")
            else:
                name = getattr(call, "name", None)
            if name:
                calls[str(name)] += 1
        if isinstance(usage, dict):
            input_tokens += _safe_int(usage.get("input_tokens"))
            output_tokens += _safe_int(usage.get("output_tokens"))
    return {
        "model": str(model_name),
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": calls.get("task", 0),
        "tool_calls": dict(sorted(calls.items())),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)
    if not report_bytes or not sources_bytes:
        raise RuntimeError("sandbox did not produce a non-empty report and sources.json")
    try:
        report_text = report_bytes.decode("utf-8")
        sources = json.loads(sources_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"sandbox output is invalid: {exc}") from exc
    if not report_text.strip():
        raise RuntimeError("sandbox report is empty")
    if not isinstance(sources, list) or not sources or not all(isinstance(item, dict) for item in sources):
        raise RuntimeError("sandbox sources.json must be a non-empty JSON array of source objects")

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(topic)
    report_path = reports_dir / f"{stem}.md"
    sources_path = reports_dir / f"{stem}.sources.json"
    meta_path = reports_dir / f"{stem}.meta.json"
    meta = {
        "topic": str(topic),
        **summarize(messages, elapsed, model_name),
        "n_sources": len(sources),
        "source_families": sorted({str(item.get("source")) for item in sources if item.get("source")}),
    }
    output_bytes = {
        report_path: report_text.encode("utf-8"),
        sources_path: (json.dumps(sources, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        meta_path: (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    }
    temporary = []
    try:
        for destination, content in output_bytes.items():
            with tempfile.NamedTemporaryFile(dir=reports_dir, prefix=f".{stem}-", suffix=".tmp", delete=False) as file:
                file.write(content)
                temporary.append((Path(file.name), destination))
        for temp_path, destination in temporary:
            os.replace(temp_path, destination)
    except Exception:
        for temp_path, _ in temporary:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return report_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    topic = str(topic or "").strip()
    if not topic:
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2

    try:
        model = make_model()
        model_name = (os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL") or
                      getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__)
        started = time.monotonic()
        with open_sandbox() as backend:
            response = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            if getattr(response, "exit_code", 0) != 0:
                raise RuntimeError(f"could not create sandbox work directories: {getattr(response, 'output', '')}")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })
            agent = build_lead_agent(backend, model)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                config={"recursion_limit": 1000},
            )
            messages = result.get("messages", []) if isinstance(result, dict) else []
            elapsed = time.monotonic() - started
            report_path = save_outputs(backend, topic, messages, elapsed, model_name)
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Report saved to {report_path}")
    return 0


def _safe_int(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
