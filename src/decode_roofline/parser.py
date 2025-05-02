import csv
import io
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_DURATION_TO_MS = {
    "nsecond": 1e-6,
    "ns": 1e-6,
    "usecond": 1e-3,
    "us": 1e-3,
    "msecond": 1.0,
    "ms": 1.0,
    "second": 1e3,
    "s": 1e3,
}
_BYTES = {
    "byte": 1,
    "bytes": 1,
    "kbyte": 1024,
    "kbytes": 1024,
    "mbyte": 1024**2,
    "mbytes": 1024**2,
    "gbyte": 1024**3,
    "gbytes": 1024**3,
}
_DURATION_METRICS = {"gpu__time_duration.sum", "gpu__time_duration.avg"}
_READ_METRICS = {"dram__bytes_read.sum"}
_WRITE_METRICS = {"dram__bytes_write.sum"}


@dataclass(frozen=True)
class NcuMeasurement:
    duration_ms: float
    dram_read_bytes: int
    dram_write_bytes: int
    kernel_instances: int
    source: str
    origin: str

    @property
    def dram_bytes(self) -> int:
        return self.dram_read_bytes + self.dram_write_bytes

    @property
    def achieved_hbm_gbps(self) -> float:
        return self.dram_bytes / self.duration_ms / 1e6


def _number(row: dict[str, str]) -> float:
    try:
        return float(row["Metric Value"].replace(",", ""))
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid Nsight Compute metric row: {row}") from error


def _unit(row: dict[str, str]) -> str:
    return row.get("Metric Unit", "").strip().lower()


def parse_ncu_csv_text(text: str, source: str = "<memory>") -> NcuMeasurement:
    rows = list(csv.DictReader(text.splitlines()))
    if not rows or "Metric Name" not in (rows[0] or {}):
        raise ValueError("expected Nsight Compute raw CSV with a Metric Name column")

    duration_ms = 0.0
    read_bytes = 0
    write_bytes = 0
    seen: set[tuple[str, str, str]] = set()
    kernels: set[str] = set()
    origins: set[str] = set()
    for row in rows:
        metric = row.get("Metric Name", "")
        if metric not in _DURATION_METRICS | _READ_METRICS | _WRITE_METRICS:
            continue
        identity = (row.get("ID", ""), row.get("Kernel Name", ""), metric)
        if identity in seen:
            continue
        seen.add(identity)
        kernels.add(identity[:2])
        origins.add(row.get("Data Origin", "ncu").strip().lower())
        value = _number(row)
        unit = _unit(row)
        if metric in _DURATION_METRICS:
            if unit not in _DURATION_TO_MS:
                raise ValueError(f"unsupported duration unit: {unit!r}")
            duration_ms += value * _DURATION_TO_MS[unit]
        else:
            if unit not in _BYTES:
                raise ValueError(f"unsupported byte unit: {unit!r}")
            if metric in _READ_METRICS:
                read_bytes += round(value * _BYTES[unit])
            else:
                write_bytes += round(value * _BYTES[unit])

    missing = []
    if not any(identity[2] in _DURATION_METRICS for identity in seen):
        missing.append("gpu__time_duration.sum")
    if not any(identity[2] in _READ_METRICS for identity in seen):
        missing.append("dram__bytes_read.sum")
    if not any(identity[2] in _WRITE_METRICS for identity in seen):
        missing.append("dram__bytes_write.sum")
    if missing:
        raise ValueError(f"Nsight Compute report lacks metrics: {', '.join(missing)}")
    if duration_ms <= 0:
        raise ValueError("reported kernel duration must be positive")
    origin = origins.pop() if len(origins) == 1 else "mixed"
    return NcuMeasurement(
        duration_ms=duration_ms,
        dram_read_bytes=read_bytes,
        dram_write_bytes=write_bytes,
        kernel_instances=len(kernels),
        source=source,
        origin=origin,
    )


def parse_ncu_csv(path: str | Path) -> NcuMeasurement:
    source = Path(path)
    return parse_ncu_csv_text(source.read_text(), source=str(source))


def parse_ncu_report(path: str | Path) -> NcuMeasurement:
    source = Path(path)
    if source.suffix == ".csv":
        return parse_ncu_csv(source)
    command = ["ncu", "--import", str(source), "--csv", "--page", "raw"]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("ncu is required to import a .ncu-rep file") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"ncu import failed: {error.stderr.strip()}") from error
    return parse_ncu_csv_text(completed.stdout, source=str(source))


def select_kernels(text: str, pattern: str) -> str:
    regex = re.compile(pattern)
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError("empty Nsight Compute CSV")
    selected = [row for row in rows if regex.search(row.get("Kernel Name", ""))]
    if not selected:
        raise ValueError(f"no kernels match {pattern!r}")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(selected)
    return buffer.getvalue()
