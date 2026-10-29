#!/usr/bin/env python3
"""
profile_env.py -- Task 0.1: Machine Telemetry Extraction

Introspects the host platform and writes results/host_profile.json containing:
  * CPU model name, microarchitecture and base frequency
  * physical core count vs. logical threads (SMT / Hyper-Threading detection)
  * L1 data, L2 and L3 cache sizes plus the cache-line size
  * OS kernel version and Python implementation details

Primary target is Windows: topology and caches come from the Win32 API
GetLogicalProcessorInformationEx (via ctypes), the CPU name and nominal clock
from the registry, with WMI as a cross-check. No admin rights or third-party
packages are required. Linux (/proc/cpuinfo + /sys) and macOS (sysctl) are
also supported. psutil and py-cpuinfo are used opportunistically if installed.

Usage (from the project root):
    python src/profile_env.py
    python src/profile_env.py --print
    python src/profile_env.py --output some/other/path.json
    python src/profile_env.py --student-id 12345678   # + results/salted_params.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import struct
import subprocess
import sys
from pathlib import Path

try:  # optional
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "host_profile.json"
SCHEMA_VERSION = 1

# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #

def _run(cmd: list[str], timeout: float = 20.0) -> str | None:
    """Run a command and return stripped stdout, or None on any failure."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _read(path) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _human_bytes(n: int | None) -> str | None:
    if not n:
        return None
    for unit, factor in (("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= factor:
            value = n / factor
            return f"{value:g} {unit}" if value != int(value) else f"{int(value)} {unit}"
    return f"{n} B"


def _parse_size(text: str | None) -> int | None:
    """'48K' / '1280K' / '32M' / '49152' -> bytes."""
    if not text:
        return None
    m = re.fullmatch(r"\s*(\d+)\s*([KMG]?)i?B?\s*", text, re.I)
    if not m:
        return None
    mult = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[m.group(2).upper()]
    return int(m.group(1)) * mult


def _count_cpu_list(text: str | None) -> int | None:
    """'0-3,8-11' -> 8"""
    if not text:
        return None
    total = 0
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            total += int(hi) - int(lo) + 1
        else:
            total += 1
    return total


def _freq_from_brand(name: str | None) -> float | None:
    """Nominal clock embedded in brand strings, e.g. '... CPU @ 2.60GHz'."""
    m = re.search(r"@\s*([\d.]+)\s*([GM])Hz", name or "", re.I)
    if not m:
        return None
    value = float(m.group(1))
    return round(value * 1000 if m.group(2).upper() == "G" else value, 1)


def _pick_first(candidates):
    """Return (value, source) for the first positive numeric candidate."""
    for value, source in candidates:
        try:
            if value is not None and float(value) > 0:
                return round(float(value), 1), source
        except (TypeError, ValueError):
            continue
    return None, None


_CPU_ID_RE = re.compile(
    r"Family\s+(\d+)\s+Model\s+(\d+)\s+Stepping\s+(\d+)(?:\s*,\s*(\S+))?", re.I)


def _parse_cpu_id(identifier: str | None):
    """'Intel64 Family 6 Model 154 Stepping 3, GenuineIntel' -> parts."""
    m = _CPU_ID_RE.search(identifier or "")
    if not m:
        return None, None, None, None
    return m.group(4), int(m.group(1)), int(m.group(2)), int(m.group(3))


_VENDOR_NAMES = {"GenuineIntel": "Intel", "AuthenticAMD": "AMD",
                 "HygonGenuine": "Hygon", "CentaurHauls": "VIA/Centaur"}

# --------------------------------------------------------------------------- #
# Microarchitecture lookup (CPUID display family / model)
# --------------------------------------------------------------------------- #

_INTEL_FAMILY6 = {
    0x2A: "Sandy Bridge", 0x2D: "Sandy Bridge-EP",
    0x3A: "Ivy Bridge", 0x3E: "Ivy Bridge-EP",
    0x3C: "Haswell", 0x3F: "Haswell-EP", 0x45: "Haswell", 0x46: "Haswell",
    0x3D: "Broadwell", 0x47: "Broadwell", 0x4F: "Broadwell-EP", 0x56: "Broadwell-DE",
    0x4E: "Skylake", 0x5E: "Skylake", 0x55: "Skylake-SP / Cascade Lake",
    0x8E: "Kaby Lake / Amber Lake / Whiskey Lake / Comet Lake (mobile)",
    0x9E: "Kaby Lake / Coffee Lake",
    0xA5: "Comet Lake", 0xA6: "Comet Lake",
    0x66: "Cannon Lake",
    0x7D: "Ice Lake", 0x7E: "Ice Lake", 0x6A: "Ice Lake-SP", 0x6C: "Ice Lake-D",
    0x8C: "Tiger Lake", 0x8D: "Tiger Lake",
    0xA7: "Rocket Lake",
    0x97: "Alder Lake", 0x9A: "Alder Lake",
    0xB7: "Raptor Lake", 0xBA: "Raptor Lake", 0xBF: "Raptor Lake",
    0xAA: "Meteor Lake", 0xAC: "Meteor Lake",
    0xBD: "Lunar Lake",
    0xC5: "Arrow Lake", 0xC6: "Arrow Lake",
    0x8F: "Sapphire Rapids", 0xCF: "Emerald Rapids",
    0xAD: "Granite Rapids", 0xAF: "Sierra Forest",
    0x5C: "Goldmont", 0x7A: "Goldmont Plus",
    0x86: "Tremont", 0x96: "Tremont", 0x9C: "Tremont",
    0xBE: "Gracemont (Alder Lake-N)",
}


def _amd_uarch(family: int, model: int) -> str | None:
    if family == 0x15:
        return "Bulldozer family"
    if family == 0x16:
        return "Jaguar / Puma"
    if family == 0x17:
        return "Zen 2" if model >= 0x30 else "Zen / Zen+"
    if family == 0x19:
        if 0x10 <= model <= 0x1F or 0x60 <= model <= 0x7F or 0xA0 <= model <= 0xAF:
            return "Zen 4"
        if 0x40 <= model <= 0x4F:
            return "Zen 3+"
        return "Zen 3"
    if family == 0x1A:
        return "Zen 5"
    return None


def guess_microarchitecture(vendor_raw, family, model, model_name) -> str:
    name = model_name or ""
    if name.startswith("Apple M"):
        return f"{name} (Apple silicon, ARM64)"
    if "Snapdragon" in name and re.search(r"\bX\b", name):
        return "Qualcomm Oryon (ARM64)"
    if family is not None and model is not None:
        guess = None
        if vendor_raw == "GenuineIntel" and family == 6:
            guess = _INTEL_FAMILY6.get(model)
        elif vendor_raw in ("AuthenticAMD", "HygonGenuine"):
            guess = _amd_uarch(family, model)
        if guess:
            return guess
        return f"unknown (family 0x{family:X}, model 0x{model:X})"
    return "unknown"

# --------------------------------------------------------------------------- #
# Cache summarisation (shared by all platforms)
# --------------------------------------------------------------------------- #

_CACHE_TYPE_NAMES = {0: "Unified", 1: "Instruction", 2: "Data", 3: "Trace"}


def _cache_key(level: int, ctype: str) -> str | None:
    ctype = ctype.lower()
    if level == 1 and ctype.startswith("data"):
        return "L1d"
    if level == 1 and ctype.startswith("instr"):
        return "L1i"
    if level >= 2 and ctype.startswith(("unified", "data")):
        return f"L{level}"
    return None


def summarize_caches(instances: list[dict]) -> dict:
    """
    instances: one dict per physical cache instance with keys
      level, type, size_bytes, line_size_bytes, associativity, logical_cpus_sharing
    Returns {"L1d": {...}, "L1i": {...}, "L2": {...}, "L3": {...}}.
    Hybrid CPUs (e.g. Intel P-cores + E-cores) can have several variants per
    level; the largest variant is reported at top level, all under 'variants'.
    """
    groups: dict[str, dict[tuple, int]] = {}
    for c in instances:
        key = _cache_key(int(c["level"]), str(c["type"]))
        if key is None or not c.get("size_bytes"):
            continue
        variant = (c["size_bytes"], c.get("line_size_bytes"),
                   c.get("associativity"), c.get("logical_cpus_sharing"))
        bucket = groups.setdefault(key, {})
        bucket[variant] = bucket.get(variant, 0) + 1

    order = {"L1d": 0, "L1i": 1, "L2": 2, "L3": 3, "L4": 4}
    out = {}
    for key in sorted(groups, key=lambda k: order.get(k, 9)):
        variants = sorted(groups[key].items(), key=lambda kv: (-kv[0][0], -kv[1]))
        entries = [{
            "size_bytes": size,
            "size_human": _human_bytes(size),
            "line_size_bytes": line,
            "associativity": assoc,
            "logical_cpus_sharing": sharing,
            "instances": count,
        } for (size, line, assoc, sharing), count in variants]
        summary = dict(entries[0])
        summary["instances"] = sum(e["instances"] for e in entries)
        summary["total_bytes"] = sum(e["size_bytes"] * e["instances"] for e in entries)
        summary["total_human"] = _human_bytes(summary["total_bytes"])
        summary["heterogeneous"] = len(entries) > 1
        if len(entries) > 1:
            summary["variants"] = entries
        out[key] = summary
    return out

# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #

_RELATION_PROCESSOR_CORE = 0
_RELATION_CACHE = 2
_RELATION_PROCESSOR_PACKAGE = 3
_RELATION_ALL = 0xFFFF
_LTP_PC_SMT = 0x1
_ERROR_INSUFFICIENT_BUFFER = 122


def parse_glpi_ex(raw: bytes, pointer_size: int) -> dict:
    """
    Parse the buffer returned by GetLogicalProcessorInformationEx(RelationAll).

    Record layout (winnt.h):
      DWORD Relationship; DWORD Size; union {...}  (union starts at offset 8)
      PROCESSOR_RELATIONSHIP: BYTE Flags; BYTE EfficiencyClass; BYTE Reserved[20];
                              WORD GroupCount; GROUP_AFFINITY GroupMask[] @ +24
      CACHE_RELATIONSHIP:     BYTE Level; BYTE Associativity; WORD LineSize;
                              DWORD CacheSize; DWORD Type; BYTE Reserved[18];
                              WORD GroupCount; GROUP_AFFINITY GroupMask[] @ +32
      GROUP_AFFINITY:         KAFFINITY Mask; WORD Group; WORD Reserved[3]
    """
    mask_fmt = "<Q" if pointer_size == 8 else "<I"
    ga_size = pointer_size + 8

    def mask_bits(base: int, count: int) -> int:
        bits = 0
        for i in range(count):
            (mask,) = struct.unpack_from(mask_fmt, raw, base + i * ga_size)
            bits += bin(mask).count("1")
        return bits

    cores, caches, packages = [], [], 0
    offset = 0
    while offset + 8 <= len(raw):
        relationship, size = struct.unpack_from("<II", raw, offset)
        body = offset + 8
        if relationship == _RELATION_PROCESSOR_CORE:
            flags, eff_class = struct.unpack_from("<BB", raw, body)
            (group_count,) = struct.unpack_from("<H", raw, body + 22)
            cores.append({
                "smt": bool(flags & _LTP_PC_SMT),
                "efficiency_class": eff_class,
                "threads": mask_bits(body + 24, max(group_count, 1)),
            })
        elif relationship == _RELATION_PROCESSOR_PACKAGE:
            packages += 1
        elif relationship == _RELATION_CACHE:
            level, assoc, line, csize, ctype = struct.unpack_from("<BBHII", raw, body)
            (group_count,) = struct.unpack_from("<H", raw, body + 30)
            caches.append({
                "level": level,
                "type": _CACHE_TYPE_NAMES.get(ctype, str(ctype)),
                "size_bytes": csize,
                "line_size_bytes": line,
                "associativity": "fully associative" if assoc == 0xFF else assoc,
                # GroupCount is 'Reserved' (0) before Windows 11 -> one mask.
                "logical_cpus_sharing": mask_bits(body + 32, max(group_count, 1)),
            })
        if size == 0:
            break
        offset += size

    by_class: dict[int, dict] = {}
    for core in cores:
        entry = by_class.setdefault(core["efficiency_class"],
                                    {"efficiency_class": core["efficiency_class"],
                                     "cores": 0, "threads": 0})
        entry["cores"] += 1
        entry["threads"] += core["threads"]

    return {
        "physical_cores": len(cores),
        "logical_processors": sum(c["threads"] for c in cores),
        "smt_capable_cores": sum(1 for c in cores if c["smt"]),
        "sockets": packages or None,
        "core_types": sorted(by_class.values(),
                             key=lambda e: -e["efficiency_class"]),
        "cache_instances": caches,
    }


def _win_topology() -> dict:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = kernel32.GetLogicalProcessorInformationEx
    fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    fn.restype = wintypes.BOOL

    length = wintypes.DWORD(0)
    if not fn(_RELATION_ALL, None, ctypes.byref(length)):
        err = ctypes.get_last_error()
        if err != _ERROR_INSUFFICIENT_BUFFER:
            raise ctypes.WinError(err)
    buf = ctypes.create_string_buffer(length.value)
    if not fn(_RELATION_ALL, buf, ctypes.byref(length)):
        raise ctypes.WinError(ctypes.get_last_error())
    return parse_glpi_ex(buf.raw[:length.value], ctypes.sizeof(ctypes.c_void_p))


def _win_registry(path: str, names: list[str]) -> dict:
    import winreg
    out = {}
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
        for name in names:
            try:
                out[name] = winreg.QueryValueEx(key, name)[0]
            except OSError:
                out[name] = None
    return out


def _win_wmi_processor() -> dict:
    script = ("Get-CimInstance Win32_Processor | Select-Object -First 1 "
              "Name,Manufacturer,MaxClockSpeed,NumberOfCores,"
              "NumberOfLogicalProcessors,L2CacheSize,L3CacheSize "
              "| ConvertTo-Json -Compress")
    out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])
    if not out:
        return {}
    try:
        data = json.loads(out)
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def probe_windows(warnings: list[str]) -> dict:
    reg = {}
    try:
        reg = _win_registry(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                            ["ProcessorNameString", "Identifier",
                             "VendorIdentifier", "~MHz"])
    except OSError as exc:
        warnings.append(f"Registry CPU query failed: {exc}")
    wmi = _win_wmi_processor()
    if not wmi:
        warnings.append("WMI (Win32_Processor) query unavailable; skipped cross-check.")

    identifier = (os.environ.get("PROCESSOR_IDENTIFIER") or reg.get("Identifier")
                  or platform.processor() or "")
    vendor_raw, family, model, stepping = _parse_cpu_id(identifier)
    vendor_raw = vendor_raw or reg.get("VendorIdentifier")
    name = re.sub(r"\s+", " ", (reg.get("ProcessorNameString") or wmi.get("Name")
                                or platform.processor() or "")).strip()

    base, base_src = _pick_first([
        (_freq_from_brand(name), "CPU brand string (nominal clock)"),
        (wmi.get("MaxClockSpeed"), "WMI Win32_Processor.MaxClockSpeed"),
        (reg.get("~MHz"), r"Registry CentralProcessor\0 '~MHz'"),
    ])

    topo = {}
    try:
        topo = _win_topology()
    except Exception as exc:  # noqa: BLE001 -- report and fall back
        warnings.append(f"GetLogicalProcessorInformationEx failed: {exc}")

    cache_instances = topo.get("cache_instances") or []
    if not cache_instances and wmi:
        warnings.append("Cache sizes taken from WMI (aggregate totals, no L1d).")
        for level, field in ((2, "L2CacheSize"), (3, "L3CacheSize")):
            kib = _to_int(wmi.get(field))
            if kib:
                cache_instances.append({"level": level, "type": "Unified",
                                        "size_bytes": kib * 1024})

    core_types = topo.get("core_types") or []
    return {
        "model_name": name or None,
        "vendor_raw": vendor_raw,
        "family": family, "model": model, "stepping": stepping,
        "base_frequency_mhz": base,
        "base_frequency_source": base_src,
        "max_frequency_mhz": None,
        "sockets": topo.get("sockets"),
        "physical_cores": topo.get("physical_cores") or _to_int(wmi.get("NumberOfCores")),
        "logical_processors": (topo.get("logical_processors")
                               or _to_int(wmi.get("NumberOfLogicalProcessors"))),
        "smt_capable_cores": topo.get("smt_capable_cores"),
        "core_types": core_types if len(core_types) > 1 else None,
        "cache_instances": cache_instances,
        "topology_source": ("Win32 GetLogicalProcessorInformationEx" if topo
                            else "WMI Win32_Processor"),
    }

# --------------------------------------------------------------------------- #
# Linux
# --------------------------------------------------------------------------- #

def probe_linux(warnings: list[str]) -> dict:
    first: dict[str, str] = {}
    pairs = set()
    for block in (_read("/proc/cpuinfo") or "").split("\n\n"):
        kv = {}
        for line in block.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                kv[k.strip()] = v.strip()
        if "processor" in kv:
            first = first or kv
            if "physical id" in kv and "core id" in kv:
                pairs.add((kv["physical id"], kv["core id"]))

    name = first.get("model name") or first.get("Hardware") or platform.processor()
    vendor_raw = first.get("vendor_id")
    family, model, stepping = (_to_int(first.get("cpu family")),
                               _to_int(first.get("model")),
                               _to_int(first.get("stepping")))

    cpu_root = Path("/sys/devices/system/cpu")
    cpu_dirs = sorted(p for p in cpu_root.glob("cpu[0-9]*") if p.is_dir())
    cores, packages = set(), set()
    for d in cpu_dirs:
        core_id = _read(d / "topology" / "core_id")
        pkg_id = _read(d / "topology" / "physical_package_id")
        if core_id is not None:
            cores.add((pkg_id, core_id))
            packages.add(pkg_id)
    if not cores and pairs:
        cores, packages = pairs, {p for p, _ in pairs}

    logical = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    online = _count_cpu_list(_read(cpu_root / "online"))
    logical = online or logical or os.cpu_count()

    # Frequencies (sysfs values are in kHz)
    base_khz = _to_int(_read(cpu_root / "cpu0/cpufreq/base_frequency"))
    max_khz = _to_int(_read(cpu_root / "cpu0/cpufreq/cpuinfo_max_freq"))
    base, base_src = _pick_first([
        (base_khz / 1000 if base_khz else None, "sysfs cpufreq/base_frequency"),
        (_freq_from_brand(name), "CPU brand string (nominal clock)"),
        (first.get("cpu MHz"), "/proc/cpuinfo 'cpu MHz' (current clock, approximate)"),
    ])

    # Caches: dedupe instances by (level, type, shared_cpu_list)
    seen, instances = set(), []
    for d in cpu_dirs:
        for idx in sorted((d / "cache").glob("index*")):
            level, ctype = _read(idx / "level"), _read(idx / "type")
            size = _parse_size(_read(idx / "size"))
            shared = _read(idx / "shared_cpu_list") or d.name
            if not (level and ctype and size):
                continue
            key = (level, ctype, shared)
            if key in seen:
                continue
            seen.add(key)
            instances.append({
                "level": int(level), "type": ctype, "size_bytes": size,
                "line_size_bytes": _to_int(_read(idx / "coherency_line_size")),
                "associativity": _to_int(_read(idx / "ways_of_associativity")),
                "logical_cpus_sharing": _count_cpu_list(shared) if "/" not in shared else 1,
            })
    if not instances:
        warnings.append("sysfs cache info unavailable; falling back to getconf.")
        instances = _getconf_caches()

    return {
        "model_name": name, "vendor_raw": vendor_raw,
        "family": family, "model": model, "stepping": stepping,
        "base_frequency_mhz": base, "base_frequency_source": base_src,
        "max_frequency_mhz": round(max_khz / 1000, 1) if max_khz else None,
        "sockets": len(packages) or None,
        "physical_cores": len(cores) or None,
        "logical_processors": logical,
        "smt_capable_cores": None, "core_types": None,
        "cache_instances": instances,
        "topology_source": "/sys/devices/system/cpu + /proc/cpuinfo",
    }


def _getconf_caches() -> list[dict]:
    out = _run(["getconf", "-a"]) or ""
    vals = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].startswith("LEVEL"):
            vals[parts[0]] = _to_int(parts[1])
    line = vals.get("LEVEL1_DCACHE_LINESIZE")
    spec = [(1, "Data", "LEVEL1_DCACHE_SIZE"), (1, "Instruction", "LEVEL1_ICACHE_SIZE"),
            (2, "Unified", "LEVEL2_CACHE_SIZE"), (3, "Unified", "LEVEL3_CACHE_SIZE")]
    return [{"level": lvl, "type": t, "size_bytes": vals[k], "line_size_bytes": line}
            for lvl, t, k in spec if vals.get(k)]

# --------------------------------------------------------------------------- #
# macOS
# --------------------------------------------------------------------------- #

def _sysctl(name: str) -> str | None:
    return _run(["sysctl", "-n", name])


def probe_macos(warnings: list[str]) -> dict:
    name = _sysctl("machdep.cpu.brand_string")
    vendor_raw = _sysctl("machdep.cpu.vendor")
    family, model, stepping = (_to_int(_sysctl("machdep.cpu.family")),
                               _to_int(_sysctl("machdep.cpu.model")),
                               _to_int(_sysctl("machdep.cpu.stepping")))
    hz = _to_int(_sysctl("hw.cpufrequency"))
    base, base_src = _pick_first([
        (_freq_from_brand(name), "CPU brand string (nominal clock)"),
        (hz / 1e6 if hz else None, "sysctl hw.cpufrequency"),
    ])
    if base is None:
        warnings.append("Base frequency is not exposed on Apple silicon.")

    # Apple silicon reports per-cluster values under hw.perflevel0 (P-cores).
    prefix = "hw.perflevel0." if _sysctl("hw.perflevel0.l1dcachesize") else "hw."
    line = _to_int(_sysctl("hw.cachelinesize"))
    instances = []
    for lvl, ctype, key in ((1, "Data", "l1dcachesize"), (1, "Instruction", "l1icachesize"),
                            (2, "Unified", "l2cachesize"), (3, "Unified", "l3cachesize")):
        size = _to_int(_sysctl(prefix + key)) or _to_int(_sysctl("hw." + key))
        if size:
            instances.append({"level": lvl, "type": ctype, "size_bytes": size,
                              "line_size_bytes": line})
    return {
        "model_name": name, "vendor_raw": vendor_raw,
        "family": family, "model": model, "stepping": stepping,
        "base_frequency_mhz": base, "base_frequency_source": base_src,
        "max_frequency_mhz": None,
        "sockets": _to_int(_sysctl("hw.packages")),
        "physical_cores": _to_int(_sysctl("hw.physicalcpu")),
        "logical_processors": _to_int(_sysctl("hw.logicalcpu")),
        "smt_capable_cores": None, "core_types": None,
        "cache_instances": instances,
        "topology_source": "sysctl",
    }

# --------------------------------------------------------------------------- #
# OS / Python / memory
# --------------------------------------------------------------------------- #

def os_info(warnings: list[str]) -> dict:
    system = platform.system()
    info = {
        "system": system,
        "release": platform.release(),
        "version": platform.version(),
        "kernel_version": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }
    if system == "Windows":
        wv = sys.getwindowsversion()
        cur = {}
        try:
            cur = _win_registry(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
                                ["UBR", "DisplayVersion", "EditionID"])
        except OSError as exc:
            warnings.append(f"Registry OS query failed: {exc}")
        ubr = cur.get("UBR")
        info.update({
            # Windows 11 still reports major version 10; build >= 22000 is Win 11.
            "product": "Windows 11" if (wv.major == 10 and wv.build >= 22000)
                       else f"Windows {platform.release()}",
            "kernel_version": f"{wv.major}.{wv.minor}.{wv.build}" + (f".{ubr}" if ubr else ""),
            "build": wv.build,
            "display_version": cur.get("DisplayVersion"),
            "edition": platform.win32_edition() or cur.get("EditionID"),
        })
    elif system == "Darwin":
        info["product"] = f"macOS {platform.mac_ver()[0]}"
    elif system == "Linux":
        try:
            info["product"] = platform.freedesktop_os_release().get("PRETTY_NAME")
        except (OSError, AttributeError):
            info["product"] = None
    return info


def python_info() -> dict:
    gil = getattr(sys, "_is_gil_enabled", None)
    packages = {}
    try:
        from importlib import metadata
        for pkg in ("numpy", "scipy", "psutil", "py-cpuinfo", "numba"):
            try:
                packages[pkg] = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                packages[pkg] = None
    except ImportError:
        pass
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "version_full": sys.version,
        "compiler": platform.python_compiler(),
        "build": list(platform.python_build()),
        "cache_tag": sys.implementation.cache_tag,
        "executable": sys.executable,
        "pointer_bits": struct.calcsize("P") * 8,
        "byteorder": sys.byteorder,
        "gil_enabled": gil() if callable(gil) else True,
        "packages": packages,
    }


def total_memory_bytes() -> int | None:
    if psutil is not None:
        return int(psutil.virtual_memory().total)
    if os.name == "nt":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys)
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None

# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def _fill_gaps(cpu: dict, warnings: list[str]) -> None:
    """Fill anything the platform probe missed from psutil / py-cpuinfo / stdlib."""
    if psutil is not None:
        if not cpu.get("physical_cores"):
            cpu["physical_cores"] = psutil.cpu_count(logical=False)
        if not cpu.get("logical_processors"):
            cpu["logical_processors"] = psutil.cpu_count(logical=True)
        if not cpu.get("base_frequency_mhz"):
            try:
                freq = psutil.cpu_freq()
                if freq and freq.max:
                    cpu["base_frequency_mhz"] = round(freq.max, 1)
                    cpu["base_frequency_source"] = "psutil.cpu_freq().max"
            except (OSError, NotImplementedError):
                pass
    if not cpu.get("model_name") or not cpu.get("cache_instances"):
        try:
            import cpuinfo  # py-cpuinfo, optional
            ci = cpuinfo.get_cpu_info()
            cpu["model_name"] = cpu.get("model_name") or ci.get("brand_raw")
            if not cpu.get("cache_instances"):
                for lvl, t, k in ((1, "Data", "l1_data_cache_size"),
                                  (2, "Unified", "l2_cache_size"),
                                  (3, "Unified", "l3_cache_size")):
                    size = ci.get(k)
                    size = size if isinstance(size, int) else _parse_size(str(size or ""))
                    if size:
                        cpu["cache_instances"].append(
                            {"level": lvl, "type": t, "size_bytes": size})
        except ImportError:
            pass
    if not cpu.get("logical_processors"):
        cpu["logical_processors"] = os.cpu_count()
    if not cpu.get("physical_cores"):
        warnings.append("Physical core count unknown; install psutil for a fallback.")


def build_profile() -> dict:
    warnings: list[str] = []
    system = platform.system()
    probes = {"Windows": probe_windows, "Linux": probe_linux, "Darwin": probe_macos}
    probe = probes.get(system)
    if probe is None:
        warnings.append(f"No dedicated probe for {system}; using generic fallbacks.")
        cpu = {"model_name": platform.processor() or None, "cache_instances": []}
    else:
        cpu = probe(warnings)
    cpu.setdefault("cache_instances", [])
    _fill_gaps(cpu, warnings)

    caches = summarize_caches(cpu.pop("cache_instances"))
    line_size = next((c["line_size_bytes"] for k in ("L1d", "L2", "L3", "L1i")
                      if (c := caches.get(k)) and c.get("line_size_bytes")), None)
    line_source = "detected"
    if line_size is None:
        line_size, line_source = 64, "assumed (x86-64 / ARM64 default)"
        warnings.append("Cache-line size not reported by OS; assuming 64 bytes.")

    physical, logical = cpu.get("physical_cores"), cpu.get("logical_processors")
    smt_enabled = bool(physical and logical and logical > physical)
    vendor_raw = cpu.pop("vendor_raw", None)

    cpu_out = {
        "model_name": cpu.get("model_name"),
        "vendor": _VENDOR_NAMES.get(vendor_raw or "", vendor_raw),
        "cpuid": {"family": cpu.get("family"), "model": cpu.get("model"),
                  "stepping": cpu.get("stepping"), "vendor_id": vendor_raw},
        "microarchitecture": guess_microarchitecture(
            vendor_raw, cpu.get("family"), cpu.get("model"), cpu.get("model_name")),
        "base_frequency_mhz": cpu.get("base_frequency_mhz"),
        "base_frequency_source": cpu.get("base_frequency_source"),
        "max_frequency_mhz": cpu.get("max_frequency_mhz"),
        "sockets": cpu.get("sockets"),
        "physical_cores": physical,
        "logical_processors": logical,
        "threads_per_core": round(logical / physical, 2) if physical and logical else None,
        "smt_enabled": smt_enabled,
        "smt_capable_cores": cpu.get("smt_capable_cores"),
        "hybrid_core_types": cpu.get("core_types"),
        "topology_source": cpu.get("topology_source"),
    }

    def size(key):
        return caches.get(key, {}).get("size_bytes")

    profile = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        # Flat summary keys (read by verify_outputs.py); details follow below.
        "cpu_model": cpu_out["model_name"],
        "microarchitecture": cpu_out["microarchitecture"],
        "base_frequency_mhz": cpu_out["base_frequency_mhz"],
        "physical_cores": physical,
        "logical_threads": logical,
        "smt_enabled": smt_enabled,
        "l1d_cache_bytes": size("L1d"),
        "l2_cache_bytes": size("L2"),
        "l3_cache_bytes": size("L3"),
        "cache_line_bytes": line_size,
        "kernel_version": None,          # filled in below from os_info
        "python_implementation": f"{platform.python_implementation()} {platform.python_version()}",
        "cpu": cpu_out,
        "cache_summary": {
            "l1d_bytes": size("L1d"),
            "l2_bytes": size("L2"),
            "l3_bytes": size("L3"),
            "l3_total_bytes": caches.get("L3", {}).get("total_bytes"),
            "cache_line_bytes": line_size,
            "cache_line_source": line_source,
            "note": "Sizes are per instance (per core / per cluster / per L3 slice group).",
        },
        "caches": caches,
        "memory": {"total_bytes": total_memory_bytes()},
        "os": os_info(warnings),
        "python": python_info(),
        "warnings": warnings,
    }
    profile["memory"]["total_human"] = _human_bytes(profile["memory"]["total_bytes"])
    profile["kernel_version"] = profile["os"]["kernel_version"]

    # Stable host fingerprint -- usable later as the salt for problem parameters.
    basis = {
        "cpu": [cpu_out["model_name"], cpu_out["microarchitecture"],
                cpu_out["physical_cores"], cpu_out["logical_processors"]],
        "cache": [size("L1d"), size("L2"), size("L3"), line_size],
        "os": [profile["os"]["system"], profile["os"]["kernel_version"]],
        "python": [profile["python"]["implementation"], profile["python"]["version"]],
    }
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()
    profile["fingerprint"] = {"sha256": digest, "salt": digest[:16], "basis": basis}
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record the host hardware/software baseline.")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT,
                        help=f"output JSON path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--print", action="store_true", help="also print the JSON to stdout")
    parser.add_argument("--student-id", metavar="ID",
                        help="8-digit Student ID; also writes results/salted_params.json")
    args = parser.parse_args(argv)

    salted = None
    if args.student_id is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from salt import get_student_params, validate_student_id
        try:
            student_id = validate_student_id(args.student_id)
        except ValueError as exc:
            parser.error(str(exc))
        salted = get_student_params(student_id)

    profile = build_profile()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    salted_path = args.output.parent / "salted_params.json"
    if salted is not None:
        salted_out = dict(salted)
        salted_out["matrix_bytes_fp64"] = salted["N"] * salted["N"] * 8
        salted_out["buffer_bytes_fp64"] = salted["buffer_elements"] * 8
        salted_out["host_fingerprint"] = profile["fingerprint"]["salt"]
        salted_out["generated_at_utc"] = profile["generated_at_utc"]
        salted_path.write_text(json.dumps(salted_out, indent=2), encoding="utf-8")

    if args.print:
        print(json.dumps(profile, indent=2))
    else:
        c, s = profile["cpu"], profile["cache_summary"]
        print(f"CPU   : {c['model_name']} [{c['microarchitecture']}]")
        print(f"Base  : {c['base_frequency_mhz']} MHz ({c['base_frequency_source']})")
        print(f"Cores : {c['physical_cores']} physical / {c['logical_processors']} logical "
              f"(SMT {'on' if c['smt_enabled'] else 'off'})")
        print(f"Cache : L1d {_human_bytes(s['l1d_bytes'])}, L2 {_human_bytes(s['l2_bytes'])}, "
              f"L3 {_human_bytes(s['l3_bytes'])}, line {s['cache_line_bytes']} B")
        print(f"OS    : {profile['os'].get('product') or profile['os']['system']} "
              f"kernel {profile['os']['kernel_version']}")
        print(f"Python: {profile['python']['implementation']} {profile['python']['version']}")
        for w in profile["warnings"]:
            print(f"warning: {w}")
        print(f"Wrote {args.output}")
    if salted is not None:
        print(f"Salted: N={salted['N']}, buffer_elements={salted['buffer_elements']}, "
              f"seed={salted['seed']}")
        print(f"Wrote {salted_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())