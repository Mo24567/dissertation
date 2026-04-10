def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length characters, cutting at a word boundary."""
    if not text:
        return text
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    boundary = truncated.rfind(" ")
    if boundary == -1:
        return truncated
    return truncated[:boundary]
