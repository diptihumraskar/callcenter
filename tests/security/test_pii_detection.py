import pytest

from src.security.pii_redactor import redact_pii

PHONE_FORMATS = [
    "555-123-4567",
    "(555) 123-4567",
    "555.123.4567",
    "+1 555-123-4567",
    "5551234567",
]

EMAIL_FORMATS = [
    "jane@co.com",
    "jane.doe@example.org",
    "j_doe123@sub.example.co.uk",
]

SSN_FORMATS = [
    "123-45-6789",
    "987-65-4320",
]

CREDIT_CARD_FORMATS = [
    "4111111111111111",
    "4111 1111 1111 1111",
    "4111-1111-1111-1111",
]


@pytest.mark.parametrize("phone", PHONE_FORMATS)
def test_phone_formats_redacted(phone):
    result = redact_pii(f"Call me at {phone} please.")
    assert result.pii_found is True
    assert "[REDACTED_PHONE]" in result.redacted_text
    assert phone not in result.redacted_text


@pytest.mark.parametrize("email", EMAIL_FORMATS)
def test_email_formats_redacted(email):
    result = redact_pii(f"My email is {email}.")
    assert result.pii_found is True
    assert "[REDACTED_EMAIL]" in result.redacted_text


@pytest.mark.parametrize("ssn", SSN_FORMATS)
def test_ssn_formats_redacted(ssn):
    result = redact_pii(f"My SSN is {ssn}")
    assert "[REDACTED_SSN]" in result.redacted_text


@pytest.mark.parametrize("card", CREDIT_CARD_FORMATS)
def test_credit_card_formats_redacted(card):
    result = redact_pii(f"Card number: {card}")
    assert "[REDACTED_CREDIT_CARD]" in result.redacted_text


def test_pii_embedded_in_conversation_text():
    text = (
        "Sure, let me pull that up. Can you confirm your SSN? "
        "Yes it's 123-45-6789 and you can email me at jane@co.com, "
        "or call 555-123-4567 if that's easier."
    )
    result = redact_pii(text)
    assert result.pii_found is True
    assert "[REDACTED_SSN]" in result.redacted_text
    assert "[REDACTED_EMAIL]" in result.redacted_text
    assert "[REDACTED_PHONE]" in result.redacted_text
    assert "123-45-6789" not in result.redacted_text


def test_no_pii_returns_unchanged_text():
    result = redact_pii("Hello, how can I help?")
    assert result.pii_found is False
    assert result.redacted_text == "Hello, how can I help?"


def test_multiple_pii_types_in_one_pass():
    result = redact_pii("Call 555-123-4567 or email jane@co.com")
    assert "[REDACTED_PHONE]" in result.redacted_text
    assert "[REDACTED_EMAIL]" in result.redacted_text
