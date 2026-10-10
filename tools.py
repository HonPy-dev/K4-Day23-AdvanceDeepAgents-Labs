"""tools.py - Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import random
import re
import threading
import time  # noqa: F401
import xml.etree.ElementTree  # noqa: F401  (arXiv answers with Atom XML)
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote, quote_plus

import httpx  # noqa: F401
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
_ARXIV_LOCK = threading.Lock()
_ARXIV_LAST_CALL = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- 1.1 Retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    attempts = max(1, int(attempts))
    for attempt in range(attempts):
        try:
            return fn()
        except (RetryableError, httpx.TransportError) as exc:
            if attempt == attempts - 1:
                raise
            retry_after = getattr(exc, "retry_after", None)
            if retry_after is not None:
                try:
                    delay = min(max(0.0, float(retry_after)), cap)
                except (TypeError, ValueError):
                    delay = _backoff_delay(base, cap, attempt)
            else:
                delay = _backoff_delay(base, cap, attempt)
            time.sleep(delay)


# ---- 1.2 arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # PSEUDO-CODE:
    #   keep only word characters of `query` -> terms; no terms -> "NO RESULTS" (do not call the network)
    #   respect arXiv etiquette: at least 3 seconds between two arXiv calls (remember the time of the last call)
    #   GET ARXIV_URL params: search_query="all:t1 AND all:t2 ...", sortBy=submittedDate, sortOrder=descending,
    #       max_results=clamp(max_results, 1, 30)           (wrap in with_retry)
    #   parse the Atom XML: each <entry> -> {id (last part of <id> after /abs/), url, published[:10], title, summary}
    #       collapse whitespace/newlines in title and summary; cut summary to ~600 chars
    #   no entries -> "NO RESULTS"; else json.dumps(records, ensure_ascii=False)
    #   any exception -> "ERROR: <type>: <message>"
    global _ARXIV_LAST_CALL
    terms = re.findall(r"[A-Za-z0-9-]+", str(query or ""))
    if not terms:
        return "NO RESULTS"
    search_query = " AND ".join(f"all:{term}" for term in terms)

    def request():
        global _ARXIV_LAST_CALL
        with _ARXIV_LOCK:
            wait = 3.0 - (time.monotonic() - _ARXIV_LAST_CALL)
            if _ARXIV_LAST_CALL and wait > 0:
                time.sleep(wait)
            _ARXIV_LAST_CALL = time.monotonic()
        response = httpx.get(
            ARXIV_URL,
            params={"search_query": search_query, "sortBy": "submittedDate", "sortOrder": "descending",
                    "max_results": min(30, max(1, int(max_results)))},
            timeout=30,
        )
        _raise_for_retryable_status(response)
        response.raise_for_status()
        return response.text

    try:
        xml_text = with_retry(request, attempts=7, base=3.0, cap=60.0)
        root = xml.etree.ElementTree.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("atom:entry", ns):
            raw_id = (entry.findtext("atom:id", default="", namespaces=ns).rsplit("/abs/", 1)[-1])
            paper_id = re.sub(r"v\d+$", "", raw_id.strip())
            if not paper_id:
                continue
            title = " ".join(entry.findtext("atom:title", default="", namespaces=ns).split())
            summary = " ".join(entry.findtext("atom:summary", default="", namespaces=ns).split())
            published = entry.findtext("atom:published", default="", namespaces=ns)[:10]
            records.append({"id": paper_id, "url": f"https://arxiv.org/abs/{paper_id}", "published": published,
                            "title": title, "summary": summary[:600]})
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # tools must return errors to the agent instead of raising
        return _error_text(exc)


# ---- 1.3 Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    # PSEUDO-CODE:
    #   GET HF_DAILY_URL params: limit (clamp 1..100) and date (only when given)      (with_retry)
    #   response = list of items {"paper": {id, title, summary, upvotes, githubRepo, githubStars, publishedAt}, ...}
    #   map every item to the record shape above (skip items without paper.id); url = https://huggingface.co/papers/<id>
    #   keyword -> keep records whose title+summary contains it (case-insensitive); sort by upvotes descending
    def request():
        params = {"limit": min(100, max(1, int(limit)))}
        if date:
            params["date"] = date
        response = httpx.get(HF_DAILY_URL, params=params, timeout=30)
        _raise_for_retryable_status(response)
        response.raise_for_status()
        return response.json()

    try:
        items = with_retry(request)
        records = []
        needle = str(keyword or "").casefold().strip()
        for item in items if isinstance(items, list) else []:
            paper = item.get("paper") or {} if isinstance(item, dict) else {}
            paper_id = str(paper.get("id") or "").strip()
            if not paper_id:
                continue
            title = _clean_text(paper.get("title") or item.get("title"))
            summary = _clean_text(paper.get("summary") or item.get("summary"))
            if needle and needle not in f"{title} {summary}".casefold():
                continue
            records.append({"id": paper_id, "url": f"https://huggingface.co/papers/{paper_id}",
                            "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
                            "title": title, "summary": summary[:600], "upvotes": _as_int(paper.get("upvotes")),
                            "github": paper.get("githubRepo"), "stars": _as_int(paper.get("githubStars"))})
        records.sort(key=lambda record: record["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error_text(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    # PSEUDO-CODE:
    #   GET HF_SEARCH_URL params: q=query, limit (clamp 1..50)                         (with_retry)
    #   same item shape as the daily endpoint; prefer paper["ai_summary"] over paper["summary"] when present
    if not str(query or "").strip():
        return "NO RESULTS"

    def request():
        response = httpx.get(HF_SEARCH_URL, params={"q": query, "limit": min(50, max(1, int(limit)))}, timeout=30)
        _raise_for_retryable_status(response)
        response.raise_for_status()
        return response.json()

    try:
        items = with_retry(request)
        records = []
        for item in items if isinstance(items, list) else []:
            paper = item.get("paper") or item if isinstance(item, dict) else {}
            paper_id = str(paper.get("id") or "").strip()
            if not paper_id:
                continue
            title = _clean_text(paper.get("title") or item.get("title"))
            summary = _clean_text(paper.get("ai_summary") or paper.get("summary") or item.get("summary"))
            records.append({"id": paper_id, "url": f"https://huggingface.co/papers/{paper_id}",
                            "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
                            "title": title, "summary": summary[:600], "upvotes": _as_int(paper.get("upvotes")),
                            "github": paper.get("githubRepo"), "stars": _as_int(paper.get("githubStars"))})
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error_text(exc)


# ---- 1.4 web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    # PSEUDO-CODE:
    #   call the MCP tool "web_search_exa" with arguments {query, objective, numResults}
    #       (objective is REQUIRED by Exa: when empty, build one from the query)
    #   see GUIDE.md part 1.4 for how to call an MCP server over plain HTTP (JSON-RPC "tools/call") and read the answer
    #   read optional env EXA_API_KEY; when present it is sent to the Exa endpoint.
    #       (see GUIDE.md 1.4 for where it goes) => the key then appears in exception text: redact it before returning "ERROR: ..."
    #   WATCH OUT: read GUIDE.md 1.4 about how Exa signals "rate limited" on the free tier, and retry on it
    query = str(query or "").strip()
    if not query:
        return "NO RESULTS"
    objective = str(objective or "").strip() or f"Find reliable research sources about: {query}"
    try:
        result = _exa_call("web_search_exa", {"query": query, "objective": objective,
                                               "numResults": min(10, max(1, int(num_results)))})
        text = "\n".join(result).strip()
        return _redact_secret(text) or "NO RESULTS"
    except Exception as exc:
        return _error_text(exc)


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        url = str(url or "").strip()
        if not url.startswith(("http://", "https://")):
            return "ERROR: ValueError: url must start with http:// or https://"
        result = _exa_call("web_fetch_exa", {"urls": [url]})
        text = "\n".join(result).strip()
        return _redact_secret(text[:12000]) if text else "NO RESULTS"
    except Exception as exc:
        return _error_text(exc)


def _raise_for_retryable_status(response):
    if response.status_code in {429, 500, 502, 503, 504}:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                retry_after = float(retry_after)
            except (TypeError, ValueError):
                try:
                    when = parsedate_to_datetime(retry_after)
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    retry_after = max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    retry_after = None
        raise RetryableError(f"HTTP {response.status_code}", retry_after=retry_after)


def _clean_text(value):
    return " ".join(str(value or "").split())


def _as_int(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _error_text(exc):
    return f"ERROR: {_redact_secret(f'{type(exc).__name__}: {exc}')}"


def _redact_secret(value):
    text = str(value)
    key = (os.getenv("EXA_API_KEY") or "").strip()
    if key:
        for spelling in {key, quote(key, safe=""), quote_plus(key, safe="")}:
            if spelling:
                text = text.replace(spelling, "[REDACTED]")
    return text


def _backoff_delay(base, cap, attempt):
    raw = max(0.0, base) * (2 ** attempt)
    jittered = raw + random.uniform(0.0, max(0.0, raw * 0.25))
    return min(max(0.0, cap), jittered)


def _exa_call(tool_name, arguments):
    api_key = (os.getenv("EXA_API_KEY") or "").strip()
    params = {"exaApiKey": api_key} if api_key else None

    def request():
        response = httpx.post(
            EXA_URL,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": tool_name, "arguments": arguments}},
            params=params,
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            timeout=60,
        )
        _raise_for_retryable_status(response)
        response.raise_for_status()
        payloads = []
        for line in response.text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                if data and data != "[DONE]":
                    payloads.append(json.loads(data))
        if not payloads:
            try:
                payloads = [response.json()]
            except ValueError as exc:
                raise ValueError("Exa returned no JSON-RPC data") from exc
        for payload in payloads:
            if "error" in payload:
                raise RuntimeError(str(payload["error"]))
            result = payload.get("result") or {}
            meta_text = json.dumps(result.get("_meta", {}), ensure_ascii=False).casefold()
            content = result.get("content") or []
            text_parts = [str(part.get("text", "")) for part in content
                          if isinstance(part, dict) and part.get("type") == "text"]
            body_text = "\n".join(text_parts)
            rate_text = re.sub(r"[_-]+", " ", meta_text + " " + body_text.casefold())
            rate_compact = re.sub(r"[^a-z0-9]+", "", meta_text)
            rate_markers = ("rate limit", "too many requests", "429", "quota exceeded",
                            "free tier limit", "free tier is limited")
            if "ratelimit" in rate_compact or any(marker in rate_text for marker in rate_markers):
                retry_after = _find_retry_after(result.get("_meta", {}))
                raise RetryableError("Exa rate limit", retry_after=retry_after)
            if result.get("isError"):
                raise RuntimeError(body_text or "Exa tool reported an error")
            if text_parts:
                return text_parts
        return []

    # Exa's free tier reports rate limits inside a HTTP 200 response.
    return with_retry(request, attempts=7, base=5.0, cap=60.0)


def _find_retry_after(meta):
    if isinstance(meta, dict):
        for key, value in meta.items():
            if "retry" in str(key).casefold() and "after" in str(key).casefold():
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
            found = _find_retry_after(value)
            if found is not None:
                return found
    elif isinstance(meta, list):
        for value in meta:
            found = _find_retry_after(value)
            if found is not None:
                return found
    return None


# ---- 1.5 Tool registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
