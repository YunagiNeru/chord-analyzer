from __future__ import annotations

import unittest
from types import SimpleNamespace

from music_agent.reference_attribution import (
    _collect_candidates,
    _grounded_sources,
)


class ReferenceAttributionTests(unittest.TestCase):
    def test_grounded_facts_are_not_duplicated_across_sources(self) -> None:
        chunks = [
            SimpleNamespace(web=SimpleNamespace(title="source-a", uri="https://a.example")),
            SimpleNamespace(web=SimpleNamespace(title="source-b", uri="https://b.example")),
            SimpleNamespace(web=SimpleNamespace(title="source-c", uri="https://c.example")),
        ]
        supports = [
            SimpleNamespace(
                segment=SimpleNamespace(text="The recording is BPM=170 and E minor."),
                grounding_chunk_indices=[0],
            ),
            SimpleNamespace(
                segment=SimpleNamespace(text="Independent listing: 170 BPM."),
                grounding_chunk_indices=[1],
            ),
            SimpleNamespace(
                segment=SimpleNamespace(text="This page lists BPM=172."),
                grounding_chunk_indices=[2],
            ),
        ]
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        grounding_chunks=chunks,
                        grounding_supports=supports,
                    )
                )
            ]
        )

        sources = _grounded_sources(response)
        self.assertEqual(sources[0].facts, ["BPM=170", "KEY=E minor"])
        self.assertEqual(sources[1].facts, ["BPM=170"])
        self.assertEqual(sources[2].facts, ["BPM=172"])

        bpms, support, keys = _collect_candidates(sources)
        self.assertEqual(bpms, [170.0])
        self.assertEqual(support, {170.0: 2, 172.0: 1})
        self.assertEqual(keys, ["E minor"])


if __name__ == "__main__":
    unittest.main()
