"""
salt.py -- Milestone 0: per-student salted problem parameters.

Each student's problem dimensions are cryptographically bound to their
8-digit Student ID via SHA-256, so every student benchmarks a slightly
different (but fully deterministic) problem size.

Usage:
    python src/salt.py 12345678
"""
import hashlib
import json
import os
import re
import sys


def get_student_params(student_id: str):
    h = int(hashlib.sha256(student_id.encode('utf-8')).hexdigest()[:8], 16)

    # Salted matrix dimension N between 1024 and 1536 (multiple of 64)
    N = 1024 + ((h % 9) * 64)

    # Salted stride benchmark buffer size: 2^24 elements (~134 MB double-precision)
    buffer_elements = 16_777_216

    # Seed for deterministic random number generation
    seed = h % 100_007

    return {"student_id": student_id, "N": N,
            "buffer_elements": buffer_elements, "seed": seed}


def validate_student_id(student_id: str) -> str:
    """Strip whitespace and require exactly 8 digits."""
    student_id = (student_id or "").strip()
    if not re.fullmatch(r"\d{8}", student_id):
        raise ValueError(f"Student ID must be exactly 8 digits, got {student_id!r}")
    return student_id


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: python {os.path.basename(__file__)} <8-digit student id>")
    print(json.dumps(get_student_params(validate_student_id(sys.argv[1])), indent=2))