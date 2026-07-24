from __future__ import annotations

import unittest

from music_agent.reference_research import (
    ReferenceResearchResult,
    _extract_bpms,
)
from music_agent.schemas import ReferenceSource


class ReferenceResearchTests(unittest.TestCase):
    def test_extracts_both_bpm_notations_and_support(self) -> None:
        candidates, support = _extract_bpms(
            "source A: BPM=170; source B: 170 BPM; unrelated cover: BPM=110"
        )
        self.assertEqual(candidates, [110.0, 170.0])
        self.assertEqual(support[170.0], 2)
        self.assertEqual(support[110.0], 1)

    def test_primary_bpm_requires_supported_candidate_list(self) -> None:
        result = ReferenceResearchResult(
            bpm_candidates=[170.0],
            bpm_support={170.0: 2},
            sources=[
                ReferenceSource(title="A", url="https://example.com/a"),
                ReferenceSource(title="B", url="https://example.com/b"),
            ],
        )
        self.assertEqual(result.primary_bpm, 170.0)


if __name__ == "__main__":
    unittest.main()
