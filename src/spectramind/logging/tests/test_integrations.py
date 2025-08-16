from spectramind.logging import MLflowLogger, WandBLogger, LoggingConfig


def test_mlflow_disabled():
    cfg = LoggingConfig(mlflow=False)
    ml = MLflowLogger(cfg)
    ml.log_metric("acc", 0.9, 1)
    assert True


def test_wandb_disabled():
    cfg = LoggingConfig(wandb=False)
    wb = WandBLogger(cfg)
    wb.log({"acc": 0.9}, step=1)
    assert True
