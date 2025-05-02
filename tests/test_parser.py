import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from decode_roofline.parser import parse_ncu_csv, parse_ncu_report, select_kernels

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_h100.csv"


def test_synthetic_fixture_aggregates_instances_and_converts_units():
    measurement = parse_ncu_csv(FIXTURE)
    assert measurement.duration_ms == pytest.approx(0.003)
    assert measurement.dram_read_bytes == 3048
    assert measurement.dram_write_bytes == 500
    assert measurement.dram_bytes == 3548
    assert measurement.kernel_instances == 2
    assert measurement.origin == "synthetic"
    assert measurement.achieved_hbm_gbps == pytest.approx(3548 / 0.003 / 1e6)


def test_duplicate_section_rows_are_counted_once():
    text = FIXTURE.read_text()
    lines = text.splitlines()
    duplicated = "\n".join([lines[0], lines[1], lines[1], *lines[2:]])
    path = FIXTURE.with_name("duplicate.csv")
    try:
        path.write_text(duplicated)
        assert parse_ncu_csv(path).duration_ms == pytest.approx(0.003)
    finally:
        path.unlink(missing_ok=True)


def test_missing_metrics_fail_instead_of_silently_zeroing():
    lines = FIXTURE.read_text().splitlines()
    path = FIXTURE.with_name("missing.csv")
    try:
        path.write_text("\n".join(line for line in lines if "bytes_write" not in line))
        with pytest.raises(ValueError, match="dram__bytes_write.sum"):
            parse_ncu_csv(path)
    finally:
        path.unlink(missing_ok=True)


def test_binary_report_is_imported_through_ncu():
    completed = subprocess.CompletedProcess([], 0, stdout=FIXTURE.read_text(), stderr="")
    with patch("decode_roofline.parser.subprocess.run", return_value=completed) as run:
        measurement = parse_ncu_report("capture.ncu-rep")
    assert measurement.origin == "synthetic"
    run.assert_called_once_with(
        ["ncu", "--import", "capture.ncu-rep", "--csv", "--page", "raw"],
        check=True,
        capture_output=True,
        text=True,
    )


def test_kernel_selection_preserves_all_metrics_for_matching_kernel():
    selected = select_kernels(FIXTURE.read_text(), "attention")
    path = FIXTURE.with_name("selected.csv")
    try:
        path.write_text(selected)
        measurement = parse_ncu_csv(path)
        assert measurement.kernel_instances == 1
        assert measurement.duration_ms == pytest.approx(0.002)
    finally:
        path.unlink(missing_ok=True)
