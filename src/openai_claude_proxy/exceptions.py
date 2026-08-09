"""Shared protocol-conversion exceptions."""


class ConversionError(ValueError):
    """Raised when an input cannot be represented by the target protocol."""
