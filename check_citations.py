"""check_citations.py - Citation validator.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_CITATION = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")
_REF_LINE = re.compile(r"^\s*\[(\d+)\](.*)$")
_URL = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)


def _numbers(group):
    found = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            first, last = map(int, span.groups())
            if last < first:
                found.extend((first, last))
            elif last - first > 200:
                found.extend((first, last))
            else:
                found.extend(range(first, last + 1))
        else:
            found.append(int(part))
    return found


def _urls(text):
    return [match.rstrip(".,;:!?)]}") for match in _URL.findall(text)]


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []
    if not isinstance(sources, list):
        return ["sources.json must be a JSON list"]
    if not sources:
        return ["no sources in sources.json"]

    by_number = {}
    seen_urls = set()
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            problems.append(f"source entry {index} must be an object")
            continue
        number = source.get("n")
        valid_number = isinstance(number, int) and not isinstance(number, bool)
        if not valid_number:
            problems.append(f"source entry {index} has a non-integer n")
        elif number in by_number:
            problems.append(f"duplicate source number [{number}]")
        else:
            by_number[number] = source

        url = source.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source [{number}] has an invalid URL")
        elif url in seen_urls:
            problems.append(f"duplicate source URL: {url}")
        else:
            seen_urls.add(url)

    report_text = str(report_text or "")
    headings = list(_REF_HEADING.finditer(report_text))
    if not headings:
        problems.append("report is missing the ## References heading")
        body = report_text
        references = ""
    else:
        body = report_text[:headings[0].start()]
        references = report_text[headings[0].end():]
        if len(headings) > 1:
            problems.append("report contains more than one ## References heading")

    cited = set()
    for index, segment in enumerate(_CODE.split(body)):
        if index % 2:
            continue
        for match in _CITATION.finditer(segment):
            cited.update(_numbers(match.group(1)))

    for number in sorted(cited - set(by_number)):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(set(by_number) - cited):
        problems.append(f"source [{number}] never cited")

    reference_entries = {}
    for line_number, line in enumerate(references.splitlines(), start=1):
        match = _REF_LINE.match(line)
        if not match:
            if line.lstrip().startswith("["):
                problems.append(f"malformed reference line {line_number}")
            continue
        number = int(match.group(1))
        reference_entries.setdefault(number, []).append(match.group(2))
        if number not in by_number:
            problems.append(f"reference [{number}] has no matching source")

        urls = _urls(match.group(2))
        if len(urls) != 1:
            problems.append(f"reference [{number}] must contain exactly one URL")
        elif number in by_number and urls[0] != by_number[number].get("url"):
            problems.append(f"reference [{number}] URL does not match sources.json")

    for number in sorted(by_number):
        count = len(reference_entries.get(number, []))
        if count == 0:
            problems.append(f"source [{number}] has no reference line")
        elif count > 1:
            problems.append(f"source [{number}] has more than one reference line")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
