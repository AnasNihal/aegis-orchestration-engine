"""Deterministic result verification and error classification."""

from .checks import VerificationReport, verify_task
from .errors import ErrorCategory, classify_error

__all__ = ["ErrorCategory", "VerificationReport", "classify_error", "verify_task"]
