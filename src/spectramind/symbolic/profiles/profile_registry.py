"""In-memory registry for loaded profiles with persistence of the active profile."""
from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

from .profile_loader import ProfileLoadResult, SymbolicProfile
from .utils import find_repo_root, safe_dump_yaml, safe_load_yaml


class ProfileRegistry:
    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self._lock = threading.RLock()
        self._repo_root = find_repo_root(repo_root)
        self._profiles: Dict[str, SymbolicProfile] = {}
        self._hashes: Dict[str, str] = {}
        self._active_id: Optional[str] = None

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    def load_from_result(self, res: ProfileLoadResult) -> None:
        with self._lock:
            self._profiles = dict(res.profiles)
            self._hashes = dict(res.hash_by_id)

    def register(self, profile: SymbolicProfile, hash_value: str) -> None:
        with self._lock:
            self._profiles[profile.id] = profile
            self._hashes[profile.id] = hash_value

    def get(self, pid: str) -> Optional[SymbolicProfile]:
        with self._lock:
            return self._profiles.get(pid)

    def list_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._profiles.keys())

    def get_hash(self, pid: str) -> Optional[str]:
        with self._lock:
            return self._hashes.get(pid)

    # Active profile persistence --------------------------------------------------
    def _active_path(self) -> Path:
        return self.repo_root / "runtime" / "active_profile.yaml"

    def set_active(self, pid: str) -> None:
        with self._lock:
            if pid not in self._profiles:
                raise KeyError(f"Profile '{pid}' not found")
            self._active_id = pid
            path = self._active_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"active_profile_id": pid, "profile": asdict(self._profiles[pid])}
            path.write_text(safe_dump_yaml(payload), encoding="utf-8")

    def get_active(self) -> Optional[str]:
        with self._lock:
            if self._active_id:
                return self._active_id
            path = self._active_path()
            if path.exists():
                data = safe_load_yaml(path)
                pid = data.get("active_profile_id")
                self._active_id = pid
                return pid
            return None


_singleton: Optional[ProfileRegistry] = None


def get_registry(repo_root: Optional[Path] = None) -> ProfileRegistry:
    global _singleton
    if _singleton is None:
        _singleton = ProfileRegistry(repo_root)
    return _singleton
