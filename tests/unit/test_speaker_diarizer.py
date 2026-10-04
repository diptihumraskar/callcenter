from src.agents.transcription import _AGENT_PATTERNS, SpeakerDiarizer


def test_customer_self_identifying_info_labeled_customer():
    d = SpeakerDiarizer()
    speaker = d.label(0.0, 5.0, "It's 123 Main Street and my phone number is 555-123-4567.")
    assert speaker == "Customer"


def test_agent_located_information_labeled_agent():
    d = SpeakerDiarizer()
    d.label(0.0, 5.0, "My phone number is 555-123-4567.")
    speaker = d.label(5.0, 8.0, "I located your information.")
    assert speaker == "Agent"


def test_my_name_is_no_longer_forces_agent():
    # A customer confirming their own identity ("my name is ...") must not
    # be force-classified as Agent - this phrase was removed from
    # _AGENT_PATTERNS because both sides say it.
    assert not _AGENT_PATTERNS.search("My name is John Smith.")


def test_no_gap_flip_when_previous_segment_is_long_but_gap_is_short():
    d = SpeakerDiarizer()
    d.label(0.0, 10.0, "Some long agent explanation with many words that goes on for a while.")
    # True silence is only 0.1s (10.1 - previous segment's END at 10.0), even
    # though the previous segment's START was 10s earlier - the gap rule must
    # key off the end timestamp, not the start.
    speaker = d.label(10.1, 15.0, "And that continues the same explanation with more detail now.")
    assert speaker == "Agent"


def test_gap_flip_when_real_silence_exceeds_threshold():
    d = SpeakerDiarizer()
    d.label(0.0, 2.0, "Some short opening line here with several words okay great.")
    speaker = d.label(4.0, 6.0, "Some short reply line here with several words okay great.")
    assert speaker == "Customer"


def test_reported_conversation_sequence_labels_correctly():
    """Regression test for the exact mislabeling reported in production."""
    d = SpeakerDiarizer()
    segments = [
        (47.0, 47.9, "Yes."),
        (48.0, 58.0, "It's 1265 north research way in Orem Utah 84097 and my phone number is 5551234567."),
        (58.0, 59.0, "Thanks John."),
        (59.0, 60.0, "I located your information."),
        (61.0, 67.0, "The newest version we have available for your vehicle is version 7.7."),
    ]
    labels = [d.label(start, end, text) for start, end, text in segments]
    assert labels[1] == "Customer"  # address + phone number is self-identifying
    assert labels[2] == "Agent"  # thanking the customer after they gave their info
    assert labels[3] == "Agent"  # "I located your information"
    assert labels[4] == "Agent"  # continues the agent's turn, no real gap
