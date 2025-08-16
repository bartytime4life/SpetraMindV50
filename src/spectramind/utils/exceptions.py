class SpectraMindError(RuntimeError):
    """Base class for SpectraMind-specific errors."""


class ConfigValidationError(SpectraMindError):
    """Configuration validation failed."""


class ReproducibilityError(SpectraMindError):
    """Reproducibility capture failed."""


class DistributedError(SpectraMindError):
    """Distributed initialization or runtime error."""


class IOValidationError(SpectraMindError):
    """I/O validation or schema mismatch error."""
