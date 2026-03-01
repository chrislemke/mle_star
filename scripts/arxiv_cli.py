#!/usr/bin/env python3
"""arXiv CLI — search and fetch academic papers for LLM consumption.

Restricted to Computer Science (cs.*) and Statistics (stat.*) categories.
Outputs JSON to stdout.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import urlencode

VALID_CATEGORIES = {"cs.AI", "cs.GT", "cs.LG", "stat.ML"}


ARXIV_API_URL = "https://export.arxiv.org/api/query"


def build_query_url(query, max_results=20, sort="relevance", category=None):
    """Build an arXiv API query URL."""
    max_results = max(1, min(max_results, 50))
    if category:
        search_query = f"all:{query} AND cat:{category}"
    else:
        search_query = f"all:{query} AND (cat:cs.* OR cat:stat.*)"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort,
        "sortOrder": "descending",
    }
    return f"{ARXIV_API_URL}?{urlencode(params)}"


ATOM_NS = "http://www.w3.org/2005/Atom"


def _normalize_whitespace(text):
    """Collapse all whitespace in text to single spaces and strip."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def parse_atom_response(xml_text):
    """Parse an Atom XML response from arXiv and return a list of paper dicts."""
    root = ET.fromstring(xml_text)
    entries = root.findall(f"{{{ATOM_NS}}}entry")
    results = []
    for entry in entries:
        # Extract arxiv_id from the id URL
        id_text = entry.findtext(f"{{{ATOM_NS}}}id", "")
        arxiv_id = id_text.rsplit("/", 1)[-1] if "/" in id_text else id_text

        title = _normalize_whitespace(entry.findtext(f"{{{ATOM_NS}}}title"))
        abstract = _normalize_whitespace(entry.findtext(f"{{{ATOM_NS}}}summary"))
        published = entry.findtext(f"{{{ATOM_NS}}}published", "")
        updated = entry.findtext(f"{{{ATOM_NS}}}updated", "")

        authors = [
            name
            for author in entry.findall(f"{{{ATOM_NS}}}author")
            if (name := author.findtext(f"{{{ATOM_NS}}}name", "").strip())
        ]

        categories = [
            term
            for cat in entry.findall(f"{{{ATOM_NS}}}category")
            if (term := cat.get("term", "").strip())
        ]

        # Find PDF link
        pdf_url = ""
        for link in entry.findall(f"{{{ATOM_NS}}}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break

        html_url = f"https://arxiv.org/html/{arxiv_id}"

        results.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "categories": categories,
                "published": published,
                "updated": updated,
                "pdf_url": pdf_url,
                "html_url": html_url,
            }
        )
    return results


class _TextExtractor(HTMLParser):
    """Extract visible text from HTML, skipping script/style tags."""

    _SKIP_TAGS = frozenset(("script", "style", "nav", "header", "footer"))

    def __init__(self):
        super().__init__()
        self._pieces = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in ("p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._pieces.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._pieces.append(data)

    def get_text(self):
        raw = "".join(self._pieces)
        lines = [line.strip() for line in raw.splitlines()]
        return "\n".join(
            line for i, line in enumerate(lines) if line or (i > 0 and lines[i - 1])
        )


def extract_text_from_html(html):
    """Extract visible text from HTML string."""
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


def _urlopen(url, timeout=30):
    """Open a URL with a User-Agent header."""
    req = urllib.request.Request(url, headers={"User-Agent": "arxiv-cli/1.0"})
    return urllib.request.urlopen(req, timeout=timeout)


def _error_exit(message):
    """Output a JSON error to stdout and exit with code 1."""
    print(json.dumps({"error": message}))
    sys.exit(1)


def cmd_search(args):
    """Handle the search subcommand."""
    if args.category and args.category not in VALID_CATEGORIES:
        valid_list = ", ".join(sorted(VALID_CATEGORIES))
        _error_exit(
            f"Invalid category '{args.category}'. Valid categories: {valid_list}"
        )

    url = build_query_url(
        args.query,
        max_results=args.max_results,
        sort=args.sort,
        category=args.category,
    )

    try:
        with _urlopen(url) as resp:
            xml_text = resp.read().decode("utf-8")
    except Exception as e:
        _error_exit(f"Network error: {e}")

    results = parse_atom_response(xml_text)
    json.dump(results, sys.stdout, indent=2)
    print()


def cmd_fetch(args):
    """Handle the fetch subcommand."""
    arxiv_id = args.arxiv_id

    # Try HTML first
    html_url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        with _urlopen(html_url, timeout=60) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        text = extract_text_from_html(html)

        # Extract title (case-insensitive)
        title = ""
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            title = match.group(1).strip()

        result = {
            "arxiv_id": arxiv_id,
            "title": title,
            "content": text,
            "source": "html",
        }
        json.dump(result, sys.stdout, indent=2)
        print()
        return
    except urllib.error.HTTPError as e:
        if e.code != 404:
            _error_exit(f"HTTP error fetching HTML: {e.code} {e.reason}")
    except Exception as e:
        _error_exit(f"Network error: {e}")

    # Fallback: fetch metadata via API
    time.sleep(3)  # rate limit
    api_url = f"{ARXIV_API_URL}?{urlencode({'id_list': arxiv_id})}"
    try:
        with _urlopen(api_url) as resp:
            xml_text = resp.read().decode("utf-8")
        papers = parse_atom_response(xml_text)
        if papers:
            paper = papers[0]
            result = {
                "arxiv_id": arxiv_id,
                "title": paper["title"],
                "content": paper["abstract"],
                "source": "abstract_only",
            }
        else:
            result = {
                "arxiv_id": arxiv_id,
                "title": "",
                "content": "",
                "source": "not_found",
            }
        json.dump(result, sys.stdout, indent=2)
        print()
    except Exception as e:
        _error_exit(f"Network error fetching metadata: {e}")


def build_parser():
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Search arXiv for CS and Statistics papers."
    )
    subparsers = parser.add_subparsers(dest="command")

    # search subcommand
    search_parser = subparsers.add_parser("search", help="Search for papers")
    search_parser.add_argument("query", help="Search query string")
    search_parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum number of results (default: 10)",
    )
    search_parser.add_argument(
        "--sort",
        choices=["relevance", "submittedDate", "lastUpdatedDate"],
        default="relevance",
        help="Sort order (default: relevance)",
    )
    search_parser.add_argument(
        "--category",
        default=None,
        help="arXiv category filter (e.g., cs.LG)",
    )
    search_parser.set_defaults(func=cmd_search)

    # fetch subcommand
    fetch_parser = subparsers.add_parser("fetch", help="Fetch a paper by ID")
    fetch_parser.add_argument("arxiv_id", help="arXiv paper ID")
    fetch_parser.set_defaults(func=cmd_fetch)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
