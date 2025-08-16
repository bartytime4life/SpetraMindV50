import shutil
from typing import Dict


def have_dvc() -> bool:
    return shutil.which("dvc") is not None


def have_lakefs() -> bool:
    return shutil.which("lakectl") is not None or shutil.which("lakefs") is not None


def dvc_status() -> Dict[str, str]:
    return {"installed": str(have_dvc()).lower()}


def lakefs_status() -> Dict[str, str]:
    return {"installed": str(have_lakefs()).lower()}
