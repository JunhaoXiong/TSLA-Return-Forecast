import pytest
import yaml

from tslaf.config import load_config


def test_default_config_loads(cfg):
    assert cfg.target_ticker == "TSLA"
    assert cfg.start == "2010-06-29" and cfg.end == "2025-12-31"
    assert cfg.auto_adjust is True
    assert cfg.known_splits == {"2020-08-31": 5.0, "2022-08-25": 3.0}
    assert cfg.train_frac == 0.7 and cfg.val_frac == 0.1


def test_invalid_split_fractions_rejected(tmp_path, cfg):
    raw = dict(cfg.raw)
    raw["split"] = {"train_frac": 0.8, "val_frac": 0.3}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(p)
