from __future__ import annotations

from pathlib import Path

from eval.run_eval import _aggregate, _load_golden_set
from ingest import extract_document


ROOT = Path(__file__).parents[1]


def test_golden_set_has_reviewed_distribution():
    items = _load_golden_set(ROOT / "eval" / "golden_set.json", None)
    assert len(items) == 30
    assert sum(item["answerable"] for item in items) == 25
    assert sum(not item["answerable"] for item in items) == 5
    assert sum(len(item["expected_sources"]) > 1 for item in items) == 5


def test_all_evaluation_formats_extract_exact_locations():
    fixture_dir = ROOT / "eval" / "fixtures"
    expected = {
        "algorithms_notes.pdf": {"Page 1", "Page 5"},
        "database_slides.pptx": {"Slide 1", "Slide 5"},
        "networking_handout.docx": {"Paragraph 1", "Paragraph 11"},
        "study_guide.txt": {"Lines 1-20", "Lines 21-40"},
    }
    for filename, required_locations in expected.items():
        path = fixture_dir / filename
        units = extract_document(path.read_bytes(), filename, "Reading Materials")
        assert required_locations <= {unit.location for unit in units}


def test_aggregate_refusal_metrics():
    rows = [
        {
            "answerable": True,
            "refused": False,
            "reference_match": True,
            "recall_at_1": True,
            "recall_at_3": True,
            "recall_at_4": True,
            "citation_valid": True,
            "citation_source_accurate": True,
            "prompt_tokens": 10,
            "total_tokens": 15,
            "end_to_end_ms": 20.0,
            "judge": None,
            "judge_error": None,
        },
        {
            "answerable": False,
            "refused": True,
            "reference_match": True,
            "recall_at_1": False,
            "recall_at_3": False,
            "recall_at_4": False,
            "citation_valid": True,
            "citation_source_accurate": None,
            "prompt_tokens": 5,
            "total_tokens": 8,
            "end_to_end_ms": 10.0,
            "judge": None,
            "judge_error": None,
        },
    ]
    metrics = _aggregate("rag", rows)
    assert metrics["recall_at_4"] == 1.0
    assert metrics["refusal_accuracy"] == 1.0
    assert metrics["false_refusal_rate"] == 0.0
