import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyzer.kestrel_analyzer.historical_index import (
    SourceChangedDuringHash,
    discover_library,
    hash_file_stably,
)


class HistoricalDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "photos"
        self.root.mkdir()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write(self, relative_path: str, content: bytes = b"photo") -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_recursively_discovers_mixed_formats_in_stable_path_order(self):
        self.write("2024/Zebra.NEF")
        self.write("2020/trip/alpha.jpg")
        self.write("2020/trip/ALPHA.CR3")
        self.write("2021/scan.tiff")
        self.write("notes.txt")

        report = discover_library(self.root, library_id="private-library")

        self.assertEqual(
            (
                "2020/trip/ALPHA.CR3",
                "2020/trip/alpha.jpg",
                "2021/scan.tiff",
                "2024/Zebra.NEF",
            ),
            tuple(asset.relative_path for asset in report.assets),
        )
        self.assertEqual(("raw", "rendered", "rendered", "raw"), tuple(asset.kind for asset in report.assets))
        self.assertEqual(4, len({asset.asset_id for asset in report.assets}))

    def test_duplicate_basenames_remain_distinct_assets(self):
        self.write("day-one/IMG_0001.CR3")
        self.write("day-two/IMG_0001.CR3")

        report = discover_library(self.root, library_id="private-library")

        self.assertEqual(2, len(report.assets))
        self.assertNotEqual(report.assets[0].asset_id, report.assets[1].asset_id)

    def test_skips_legacy_state_and_symlinks_without_following_them(self):
        self.write(".kingfisher/cache/hidden.jpg")
        target = self.write("outside.jpg")
        link = self.root / "linked.CR3"
        try:
            link.symlink_to(target)
        except (NotImplementedError, OSError):
            self.skipTest("symlinks are unavailable on this platform")

        report = discover_library(self.root, library_id="private-library")

        self.assertEqual(("outside.jpg",), tuple(asset.relative_path for asset in report.assets))
        self.assertEqual(("symlink_skipped",), tuple(item.code for item in report.diagnostics))

    def test_asset_identity_is_path_based_but_not_mount_path_based(self):
        first = self.write("nested/image.ARW")
        other_root = Path(self.temporary_directory.name) / "remounted"
        (other_root / "nested").mkdir(parents=True)
        (other_root / "nested/image.ARW").write_bytes(first.read_bytes())

        first_asset = discover_library(self.root, library_id="private-library").assets[0]
        remounted_asset = discover_library(other_root, library_id="private-library").assets[0]
        other_library_asset = discover_library(other_root, library_id="another-library").assets[0]

        self.assertEqual(first_asset.asset_id, remounted_asset.asset_id)
        self.assertNotEqual(first_asset.asset_id, other_library_asset.asset_id)

    def test_stable_hash_records_full_sha256_and_stat_signature(self):
        path = self.write("bird.CR3", b"a complete source image")

        fingerprint = hash_file_stably(path)

        self.assertEqual("sha256", fingerprint.algorithm)
        self.assertEqual(64, len(fingerprint.content_digest))
        self.assertEqual(path.stat().st_size, fingerprint.byte_size)
        self.assertEqual(path.stat().st_mtime_ns, fingerprint.mtime_ns)

    def test_hash_rejects_a_file_that_changes_during_read(self):
        path = self.write("changing.CR3", b"before")
        original_stat = os.stat
        calls = 0

        def changing_stat(candidate, *args, **kwargs):
            nonlocal calls
            result = original_stat(candidate, *args, **kwargs)
            calls += 1
            if calls == 2:
                path.write_bytes(b"after-and-longer")
                return original_stat(candidate, *args, **kwargs)
            return result

        with self.assertRaises(SourceChangedDuringHash):
            hash_file_stably(path, stat_function=changing_stat)

    def test_hash_rejects_regular_file_replacement_before_descriptor_open(self):
        path = self.write("replaced.CR3", b"original")
        replacement = self.write("replacement.tmp", b"replacement")
        original_open = os.open

        def replace_then_open(candidate, flags):
            os.replace(replacement, path)
            return original_open(candidate, flags)

        with patch(
            "analyzer.kestrel_analyzer.historical_index.os.open",
            side_effect=replace_then_open,
        ):
            with self.assertRaises(SourceChangedDuringHash):
                hash_file_stably(path)


if __name__ == "__main__":
    unittest.main()
