import pytest

from ask_my_docs.generation.answer import CitationValidationError, validate_citations
from ask_my_docs.models import CitationRef


def test_valid_citations():
    validate_citations(
        "Renewal is 30 days [1].",
        [CitationRef(id=1, quote="30 days")],
        num_context_blocks=2,
    )


def test_rejects_out_of_range():
    with pytest.raises(CitationValidationError):
        validate_citations("x [3]", [CitationRef(id=3)], num_context_blocks=2)
