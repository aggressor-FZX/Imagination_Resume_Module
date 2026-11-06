"""Guard the exact input used for the paired two-mode comparison."""

import hashlib
from pathlib import Path

import pytest


FIXTURE = Path(__file__).parent / "fixtures" / "bridge_opportunity"
FROZEN = {
    "candidate-main.docx": "7883ee26ab0dc57f8493be8856317d84e3a5c9bed3a48f123715347c9ef7a2b9",
    "candidate-supplement.docx": "d2263b6311f501da6e925ab401cb279ab4ed2d72ec53b666f11a40548bacd8b9",
    "candidate-main.txt": "d3c2a8e0634d3035b5dd3767d7d8469b7e2934efaf325901def3823106988732",
    "candidate-supplement.txt": "5e1e4b31a492eda862f3d15915b563717b0b44a5a50775c3897dc9531fbd6284",
    "intended-job-description.txt": "88de858c90a31e4f4955c23ad31d059179b8e47ff6caa374577a4cd348ef26d1",
}


@pytest.mark.parametrize("name,expected", FROZEN.items())
def test_frozen_comparison_input(name, expected):
    assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == expected