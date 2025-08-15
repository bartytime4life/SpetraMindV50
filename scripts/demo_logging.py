import random, time
from pathlib import Path
from spectramind.logging.logger import RunLogger
from spectramind.telemetry.events import Event

if __name__ == "__main__":
    rl = RunLogger()
    log = rl.get("spectramind.demo")
    for step in range(5):
        loss = 1.0 / (step + 1) + random.random()*0.01
        Event(kind="train", phase="step", step=step, metrics={"loss": loss}).emit(log)
        time.sleep(0.2)
    log.info("demo complete")

