"""Config loading: defaults and the `base:` override mechanism used by configs/colab.yaml."""
from badminton_analysis.config import REPO_ROOT, load_config


def test_default_config_has_required_sections():
    cfg = load_config()
    for key in ("paths", "video", "frame_shift", "pose", "court", "hits", "shots"):
        assert key in cfg.raw
    assert cfg.shift("player_bbox") == 1 and cfg.fps == 30


def test_colab_config_only_overrides_paths():
    base = load_config()
    colab = load_config(REPO_ROOT / "configs" / "colab.yaml")
    assert str(colab.bfmd_root) == "/content/BFMD_data"
    assert str(colab.output_dir).startswith("/content/drive/MyDrive/badminton")
    for key in ("frame_shift", "pose", "court", "hits", "shots", "video"):
        assert colab.raw[key] == base.raw[key]


def test_base_override_deep_merges(tmp_path):
    (tmp_path / "a.yaml").write_text("paths: {x: 1, y: 2}\nother: {k: 1}\n")
    (tmp_path / "b.yaml").write_text("base: a.yaml\npaths: {y: 3}\n")
    raw = load_config(tmp_path / "b.yaml").raw
    assert raw == {"paths": {"x": 1, "y": 3}, "other": {"k": 1}}
