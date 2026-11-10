import re

_BAD_PHRASE_RE = re.compile(
    r"ignore previous instructions|system prompt|tool override",
    re.IGNORECASE,
)


def sanitize(text: str, escape_xml: bool = True) -> str:
    if not text:
        return ""
    # Case-insensitive prompt injection protection
    text = _BAD_PHRASE_RE.sub("[REDACTED]", text)
    if escape_xml:
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text


def sanitize_chunks(chunks: list):
    for c in chunks:
        c.content = sanitize(c.content, escape_xml=False)
        if c.summary:
            c.summary = sanitize(c.summary, escape_xml=False)
