import unittest

from subtitle_formats import (
    format_timestamp_srt,
    format_timestamp_vtt,
    segments_to_srt,
    segments_to_vtt,
)


class TestOutputFormats(unittest.TestCase):
    def test_timestamp_formatting(self):
        self.assertEqual(format_timestamp_srt(0), "00:00:00,000")
        self.assertEqual(format_timestamp_vtt(0), "00:00:00.000")
        self.assertEqual(format_timestamp_srt(3661.5), "01:01:01,500")
        self.assertEqual(format_timestamp_vtt(3661.5), "01:01:01.500")

    def test_segments_to_srt(self):
        segments = [
            {"start": 0.0, "end": 1.234, "text": "Hello"},
            {"start": 1.5, "end": 2.0, "text": "World"},
        ]
        srt = segments_to_srt(segments).strip()
        self.assertIn("1\n00:00:00,000 --> 00:00:01,234\nHello", srt)
        self.assertIn("2\n00:00:01,500 --> 00:00:02,000\nWorld", srt)

    def test_segments_to_vtt(self):
        segments = [{"start": 0.0, "end": 1.234, "text": "Hello"}]
        vtt = segments_to_vtt(segments).strip()
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertIn("00:00:00.000 --> 00:00:01.234", vtt)
        self.assertIn("Hello", vtt)


if __name__ == "__main__":
    unittest.main()
