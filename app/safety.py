"""PHI redaction gate + banned language scan for Pose-Not-Pixels.

Reuses the pill-tray safety pattern:

1. Redaction. Patient labels / free text pass through ``redact_phi``
   before storage or logging. Logs carry kinds only, never values.
2. Banned language. Outputs are screening pattern language ONLY.
   They must never contain a diagnosis verdict, a treatment
   recommendation, or "you should ..." medical advice. ``find_banned_language``
   scans every product-facing string so tests fail the build if banned
   language ever slips into a summary, flag note, or template.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("pose-not-pixels.safety")

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("DOB", re.compile(
        r"\b(?:DOB|D\.O\.B\.|Date of Birth)\s*[:.]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        re.IGNORECASE)),
    ("MRN", re.compile(
        r"\b(?:MRN|Medical Record|Account|Acct)\s*#?\s*[:.]?\s*[\w-]{3,}\b",
        re.IGNORECASE)),
    ("MEMBER_ID", re.compile(
        r"\b(?:Member|Subscriber)\s*(?:ID|#|No\.?)?\s*[:.]?\s*[A-Z0-9][\w-]{4,}\b",
        re.IGNORECASE)),
    ("PHONE", re.compile(r"(?<!\d)(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}(?!\d)")),
    ("ADDRESS", re.compile(
        r"\b\d{1,5}\s+[A-Z][a-zA-Z]+\s+"
        r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Drive|Dr|Ln|Lane|Way|Ct|Court)\b.*",
        re.IGNORECASE)),
    ("NAME", re.compile(
        r"\b(?:Patient(?: Name)?|Pt\.?|Name)\s*:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})")),
]


def redact_phi(text: str) -> tuple[str, dict[str, int]]:
    report: dict[str, int] = {}
    redacted = text or ""
    for kind, pattern in _PATTERNS:
        redacted, n = pattern.subn(f"[{kind}]", redacted)
        if n:
            report[kind] = report.get(kind, 0) + n
    if report:
        logger.info("phi_redacted", extra={"phi_kinds": sorted(report.keys()),
                                           "phi_counts": report})
    return redacted, report


def looks_like_phi(text: str) -> list[str]:
    return [kind for kind, pattern in _PATTERNS if pattern.search(text or "")]


#: Language that turns a screening pattern into a verdict or advice.
#: Product-facing strings must never contain these (case-insensitive).
BANNED_PHRASES: list[str] = [
    "diagnosed",
    "you have osteoarthritis",
    "you have arthritis",
    "you have knee arthritis",
    "you have hip arthritis",
    "you should take",
    "you should do",
    "you should start",
    "you should stop",
    "you should avoid",
    "take this medicine",
    "take this medication",
    "stop taking",
    "start taking",
    "safe to take",
    "you are at high risk of falling",  # risk verdict; use threshold pattern instead
    "will fall",
    "has osteoarthritis",
    "has arthritis",
]


def find_banned_language(text: str) -> list[str]:
    low = (text or "").lower()
    return [p for p in BANNED_PHRASES if p in low]
