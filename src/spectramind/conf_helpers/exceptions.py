# SPDX-License-Identifier: MIT

"""Custom exception hierarchy for conf_helpers."""

class ConfHelpersError(Exception):
    """Base exception for configuration helpers."""


class ConfigValidationError(ConfHelpersError):
    """Raised when config validation fails."""


class OverrideLoadError(ConfHelpersError):
    """Raised when overrides cannot be loaded or merged."""


class HydraLoadError(ConfHelpersError):
    """Raised when Hydra fails to compose a config."""


class SchemaExportError(ConfHelpersError):
    """Raised when exporting schemas fails."""
