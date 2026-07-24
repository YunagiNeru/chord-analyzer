from __future__ import annotations

import unittest

from music_agent.media_source import YouTubeMediaSource


class MediaSourceTests(unittest.TestCase):
    def test_youtube_clip_part(self) -> None:
        source = YouTubeMediaSource("https://www.youtube.com/watch?v=9QLT1Aw_45s")
        part = source.part(10.0, 20.0)
        self.assertIsNotNone(part.file_data)
        self.assertEqual(part.file_data.file_uri, source.url)
        self.assertIsNotNone(part.video_metadata)


if __name__ == "__main__":
    unittest.main()
