from project01.config.settings import ROOT_DIR, CONFIG_PATH


def test_project_root_and_dataset_config_resolve_inside_project():
    assert (ROOT_DIR / "pyproject.toml").is_file()
    assert (ROOT_DIR / "src" / "project01").is_dir()
    assert CONFIG_PATH == ROOT_DIR / "config_datasets.yaml"
    assert CONFIG_PATH.is_file()
