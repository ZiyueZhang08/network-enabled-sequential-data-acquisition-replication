from __future__ import annotations

import csv
import hashlib
import re
import struct
import tempfile
import sys
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import build_figure4a_submission_bundle as bundle


def read_latest_metadata() -> dict[str, str]:
    with bundle.OUT_LATEST_CSV.open(newline="", encoding="utf-8-sig") as file:
        return {row["item"]: row["value"] for row in csv.DictReader(file)}


class Figure4aSubmissionBundleTests(unittest.TestCase):
    def test_bundle_file_list_covers_submission_surface(self) -> None:
        files = set(bundle.BUNDLE_FILES)

        self.assertEqual(len(files), 17)
        self.assertIn("reports/figure4a_submission_packet.md", files)
        self.assertIn("reports/author_source_request_email.md", files)
        self.assertIn("reports/figure4a_artifact_checksums.md", files)
        self.assertIn("outputs/tables/figure4a_final_package_summary.csv", files)
        self.assertIn(
            "outputs/figures/movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds_w_rmse.png",
            files,
        )
        self.assertIn(
            "outputs/figures/movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds_cumulative_reward.png",
            files,
        )

    def test_bundle_sources_exist_before_packaging(self) -> None:
        rows = bundle.bundle_rows()
        missing = [row["path"] for row in rows if row["exists"] != "true"]

        self.assertFalse(missing, missing)
        self.assertTrue(all(row["sha256"] for row in rows))

    def test_latest_bundle_metadata_targets_are_stable(self) -> None:
        self.assertEqual(
            bundle.OUT_LATEST_CSV.relative_to(bundle.ROOT).as_posix(),
            "outputs/tables/figure4a_submission_bundle_latest.csv",
        )
        self.assertEqual(
            bundle.OUT_LATEST_REPORT.relative_to(bundle.ROOT).as_posix(),
            "reports/figure4a_submission_bundle_latest.md",
        )

    def test_latest_bundle_pointer_uses_highest_numbered_zip(self) -> None:
        metadata = read_latest_metadata()
        zip_path = bundle.ROOT / metadata["zip_path"]
        candidates = list(bundle.OUT_DIR.glob("figure4a_submission_bundle_*_*.zip"))

        def bundle_index(path: Path) -> int:
            match = re.search(r"_(\d{3})\.zip$", path.name)
            self.assertIsNotNone(match, path.name)
            return int(match.group(1))

        highest_zip = max(candidates, key=bundle_index)
        report = bundle.OUT_LATEST_REPORT.read_text(encoding="utf-8-sig")
        report_zip_refs = set(
            re.findall(r"figure4a_submission_bundle_\d{8}_\d{3}\.zip", report)
        )

        self.assertEqual(highest_zip, zip_path)
        self.assertEqual({zip_path.name}, report_zip_refs)
        self.assertIn(metadata["zip_sha256"], report)
        self.assertIn(f"- Entry count: `{metadata['entry_count']}`", report)

    def test_latest_bundle_zip_matches_internal_manifest(self) -> None:
        metadata = read_latest_metadata()
        zip_path = bundle.ROOT / metadata["zip_path"]

        self.assertTrue(zip_path.exists(), zip_path)
        self.assertEqual(bundle.sha256(zip_path), metadata["zip_sha256"])

        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
            manifest = list(
                csv.DictReader(
                    archive.read("bundle_manifest.csv")
                    .decode("utf-8-sig")
                    .splitlines()
                )
            )
            failures = []
            for row in manifest:
                name = row["path"]
                if name not in names:
                    failures.append(f"missing:{name}")
                    continue
                data = archive.read(name)
                digest = hashlib.sha256(data).hexdigest()
                if str(len(data)) != row["bytes"]:
                    failures.append(f"bytes:{name}")
                if digest != row["sha256"]:
                    failures.append(f"sha256:{name}")

        self.assertEqual(int(metadata["entry_count"]), len(names))
        self.assertEqual({"bundle_manifest.csv"}, names - {row["path"] for row in manifest})
        self.assertFalse(failures, failures)

    def test_latest_bundle_matches_current_packaged_sources(self) -> None:
        metadata = read_latest_metadata()
        zip_path = bundle.ROOT / metadata["zip_path"]

        with zipfile.ZipFile(zip_path) as archive:
            manifest = list(
                csv.DictReader(
                    archive.read("bundle_manifest.csv")
                    .decode("utf-8-sig")
                    .splitlines()
                )
            )

        failures = []
        for row in manifest:
            if row["path"] == "bundle_manifest.csv":
                continue
            disk_path = bundle.ROOT / row["path"]
            if not disk_path.exists():
                failures.append(f"missing:{row['path']}")
                continue
            data = disk_path.read_bytes()
            if str(len(data)) != row["bytes"]:
                failures.append(f"bytes:{row['path']}")
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                failures.append(f"sha256:{row['path']}")

        self.assertFalse(failures, failures)

    def test_latest_bundle_report_matches_zip_manifest(self) -> None:
        metadata = read_latest_metadata()
        zip_path = bundle.ROOT / metadata["zip_path"]
        row_pattern = re.compile(
            r"^\| `(?P<path>[^`]+)` \| (?P<exists>true|false) \| "
            r"(?P<bytes>\d+) \| `(?P<sha256>[0-9a-f]*)` \|$"
        )
        report = bundle.OUT_LATEST_REPORT.read_text(encoding="utf-8-sig")

        with zipfile.ZipFile(zip_path) as archive:
            zip_manifest = {
                row["path"]: row
                for row in csv.DictReader(
                    archive.read("bundle_manifest.csv")
                    .decode("utf-8-sig")
                    .splitlines()
                )
            }

        report_manifest = {}
        for line in report.splitlines():
            match = row_pattern.match(line)
            if match:
                report_manifest[match.group("path")] = match.groupdict()

        self.assertIn(f"- Zip: `{metadata['zip_path']}`", report)
        self.assertIn(f"- Entry count: `{metadata['entry_count']}`", report)
        self.assertIn(f"- Zip bytes: `{metadata['zip_bytes']}`", report)
        self.assertIn(f"- Zip SHA256: `{metadata['zip_sha256']}`", report)
        self.assertEqual("18", metadata["entry_count"])
        self.assertEqual(set(zip_manifest) | {"bundle_manifest.csv"}, set(report_manifest))
        for path, row in zip_manifest.items():
            self.assertEqual(row, report_manifest[path])
        self.assertEqual("true", report_manifest["bundle_manifest.csv"]["exists"])
        self.assertTrue(report_manifest["bundle_manifest.csv"]["sha256"])

    def test_latest_bundle_readiness_audit_matches_packaged_files(self) -> None:
        metadata = read_latest_metadata()
        zip_path = bundle.ROOT / metadata["zip_path"]

        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
            manifest = {
                row["path"]: row
                for row in csv.DictReader(
                    archive.read("bundle_manifest.csv")
                    .decode("utf-8-sig")
                    .splitlines()
                )
            }
            readiness_rows = {
                row["path"].replace("\\", "/"): row
                for row in csv.DictReader(
                    archive.read("outputs/tables/figure4a_readiness_audit.csv")
                    .decode("utf-8-sig")
                    .splitlines()
                )
            }
            failures = []
            for path, row in manifest.items():
                if path == "bundle_manifest.csv":
                    continue
                if path not in readiness_rows:
                    continue
                if path not in names:
                    failures.append(f"missing:{path}")
                    continue
                actual_bytes = len(archive.read(path))
                if str(actual_bytes) != readiness_rows[path]["bytes"]:
                    failures.append(
                        f"bytes:{path}:audit={readiness_rows[path]['bytes']}:zip={actual_bytes}"
                    )

        self.assertFalse(failures, failures)

    def test_latest_bundle_artifact_checksums_match_packaged_files(self) -> None:
        metadata = read_latest_metadata()
        zip_path = bundle.ROOT / metadata["zip_path"]
        row_pattern = re.compile(
            r"^\| `(?P<path>[^`]+)` \| (?P<exists>true|false) \| "
            r"(?P<bytes>\d+) \| `(?P<sha256>[0-9a-f]*)` \|$"
        )

        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
            text = archive.read("reports/figure4a_artifact_checksums.md").decode(
                "utf-8-sig"
            )
            checksum_rows = {}
            for line in text.splitlines():
                match = row_pattern.match(line)
                if match:
                    checksum_rows[match.group("path").replace("\\", "/")] = (
                        match.groupdict()
                    )

            failures = []
            matched = 0
            for name in names:
                if name == "bundle_manifest.csv":
                    continue
                if name not in checksum_rows:
                    continue
                data = archive.read(name)
                matched += 1
                if str(len(data)) != checksum_rows[name]["bytes"]:
                    failures.append(f"bytes:{name}")
                if hashlib.sha256(data).hexdigest() != checksum_rows[name]["sha256"]:
                    failures.append(f"sha256:{name}")

        self.assertGreaterEqual(matched, 10)
        self.assertFalse(failures, failures)

    def test_artifact_checksums_exclude_volatile_latest_pointer(self) -> None:
        report = (bundle.REPORT_DIR / "figure4a_artifact_checksums.md").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("reports\\figure4a_submission_bundle_latest.md", report)
        self.assertNotIn("outputs\\tables\\figure4a_submission_bundle_latest.csv", report)

    def test_latest_bundle_contains_actionable_author_source_request(self) -> None:
        metadata = read_latest_metadata()
        zip_path = bundle.ROOT / metadata["zip_path"]

        with zipfile.ZipFile(zip_path) as archive:
            report = archive.read("reports/author_source_request_email.md").decode(
                "utf-8-sig"
            )
            rows = {
                row["item"]: row
                for row in csv.DictReader(
                    archive.read("outputs/tables/author_source_request_email.csv")
                    .decode("utf-8-sig")
                    .splitlines()
                )
            }
            body = rows["email_body"]["content"]

        self.assertIn("Status: `ready_to_send`", report)
        self.assertEqual(rows["execution_decision"]["status"], "ready_to_send")
        self.assertEqual(rows["p0_request_count"]["content"], "4")
        self.assertEqual(rows["p1_request_count"]["content"], "1")
        for expected in [
            "Linear-minus-DU gap 0.0207",
            "79.5%",
            "Exact Figure 4a w-RMSE definition",
            "MovieLens split and initialization protocol",
            "DU-Net observational entropy and DLS implementation",
            "Linear UCB feature/update protocol",
            "seeds 0,1,2",
            "do not run another local proxy sweep",
        ]:
            self.assertIn(expected, report + body)

    def test_latest_bundle_extracts_as_self_contained_handoff(self) -> None:
        metadata = read_latest_metadata()
        zip_path = bundle.ROOT / metadata["zip_path"]
        required_paths = [
            "reports/figure4a_submission_packet.md",
            "reports/figure4a_final_package.md",
            "reports/figure4a_final_delivery_zh.md",
            "reports/author_source_request_email.md",
            "outputs/figures/movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds_w_rmse.png",
            "outputs/figures/movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds_cumulative_reward.png",
        ]

        with tempfile.TemporaryDirectory(prefix="figure4a_bundle_extract_") as temp:
            extract_root = Path(temp)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_root)

            with (extract_root / "bundle_manifest.csv").open(
                newline="", encoding="utf-8-sig"
            ) as file:
                manifest = list(csv.DictReader(file))

            failures = []
            for row in manifest:
                extracted = extract_root / row["path"]
                if not extracted.exists():
                    failures.append(f"missing:{row['path']}")
                    continue
                data = extracted.read_bytes()
                if str(len(data)) != row["bytes"]:
                    failures.append(f"bytes:{row['path']}")
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    failures.append(f"sha256:{row['path']}")

            for path in required_paths:
                self.assertTrue((extract_root / path).exists(), path)

            packet = (
                extract_root / "reports/figure4a_submission_packet.md"
            ).read_text(encoding="utf-8-sig")
            email = (
                extract_root / "reports/author_source_request_email.md"
            ).read_text(encoding="utf-8-sig")

        self.assertEqual(17, len(manifest))
        self.assertFalse(failures, failures)
        self.assertIn("ready_for_caveated_submission", packet)
        self.assertIn("0.0207", packet)
        self.assertIn("79.5%", packet)
        self.assertIn("MF-centered Linear UCB", packet)
        self.assertIn("Status: `ready_to_send`", email)
        self.assertIn("seeds 0,1,2", email)

    def test_latest_bundle_figures_are_valid_pngs(self) -> None:
        metadata = read_latest_metadata()
        zip_path = bundle.ROOT / metadata["zip_path"]
        figures = [
            "outputs/figures/movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds_w_rmse.png",
            "outputs/figures/movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds_cumulative_reward.png",
        ]

        with zipfile.ZipFile(zip_path) as archive:
            for figure in figures:
                data = archive.read(figure)
                width, height = struct.unpack(">II", data[16:24])

                self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"), figure)
                self.assertGreater(len(data), 50_000, figure)
                self.assertGreaterEqual(width, 1200, figure)
                self.assertGreaterEqual(height, 700, figure)


if __name__ == "__main__":
    unittest.main()
