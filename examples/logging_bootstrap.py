import json
import logging
import logging.handlers
import os
import subprocess
import time
from pathlib import Path


def git_hash():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "NA"


def init_logging(  # noqa: PLR0913
    console_level="INFO",
    file_dir="var/logs",
    rotate_mb=50,
    keep=10,
    jsonl_path="var/events/events.jsonl",
    run="NA",
    profile="NA",
):
    Path(file_dir).mkdir(parents=True, exist_ok=True)
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, console_level))
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(
        filename=str(Path(file_dir) / "app.log"), maxBytes=rotate_mb * 1024 * 1024, backupCount=keep
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(fh)

    class Jsonl(logging.Handler):
        def __init__(self, path):
            super().__init__(logging.INFO)
            self.path = path

        def emit(self, record):
            d = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "level": record.levelname,
                "name": record.name,
                "msg": record.getMessage(),
                "run": run,
                "git": git_hash(),
                "profile": profile,
            }
            with open(self.path, "a") as f:
                f.write(json.dumps(d) + "\n")

    jh = Jsonl(jsonl_path)
    logger.addHandler(jh)
    return logger


if __name__ == "__main__":
    log = init_logging(
        run=os.getenv("SMV50_RUN", "ad-hoc"), profile=os.getenv("SMV50_PROFILE", "unknown")
    )
    log.info("jsonl + rotating file logging online.")
