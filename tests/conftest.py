import sys
import importlib.util
from pathlib import Path

# Ensure project root and src are on sys.path so `import spectramind` works during tests
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

# Force import of the package implementation rather than the CLI stub at repo root
spec = importlib.util.spec_from_file_location("spectramind", SRC / "spectramind" / "__init__.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # type: ignore[arg-type]
sys.modules["spectramind"] = module
