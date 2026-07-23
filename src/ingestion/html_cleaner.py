"""
Clean raw APP-350 privacy-policy HTML into normalised text while
preserving heading structure (needed by the structural/semantic chunkers)
and reading order.

Design notes (for your Methodology chapter, in your own words):
- We strip <script>, <style>, <nav>, <footer>, <header> boilerplate.
- We keep heading tags (h1-h4) as explicit markers in the text stream
  ("\n## <heading text>\n") so downstream chunkers can detect section
  boundaries without re-parsing HTML.
- We collapse whitespace but do not alter wording, since altering wording
  would compromise the ground-truth comparison against APP-350 annotations.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from bs4 import BeautifulSoup

HEADING_TAGS = ["h1", "h2", "h3", "h4"]
STRIP_TAGS = ["script", "style", "nav", "noscript", "svg", "iframe"]


@dataclass
class CleanedPolicy:
    policy_id: str
    text: str          # full cleaned text with heading markers
    n_headings: int
    n_chars: int


def clean_html(policy_id: str, raw_html: str) -> CleanedPolicy:
    soup = BeautifulSoup(raw_html, "lxml")

    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for tag in soup.find_all(HEADING_TAGS):
        heading_text = tag.get_text(" ", strip=True)
        if heading_text:
            tag.replace_with(f"\n\n## {heading_text}\n")

    text = soup.get_text("\n")
    text = _normalise_whitespace(text)
    n_headings = text.count("\n## ")

    return CleanedPolicy(
        policy_id=policy_id,
        text=text,
        n_headings=n_headings,
        n_chars=len(text),
    )


def _normalise_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
