from src.utils.text_utils import truncate_text


def test_short_text_unchanged():
    text = "Hello world"
    assert truncate_text(text, 20) == text


def test_empty_returns_empty():
    assert truncate_text("", 10) == ""


def test_truncation_at_word_boundary():
    text = "Hello world foo bar baz"
    result = truncate_text(text, 14)
    # text[:14] == "Hello world fo" -> rfind(' ') == 11 -> "Hello world"
    assert result == "Hello world"
    assert "fo" not in result


def test_exact_length_not_truncated():
    text = "Hello world"
    assert truncate_text(text, len(text)) == text


def test_result_shorter_than_original():
    text = "Hello world foo bar baz"
    result = truncate_text(text, 10)
    assert len(result) < len(text)
