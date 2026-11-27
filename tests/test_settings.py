from project01.config.settings import ROOT_DIR, CONFIG_PATH


def test_project_root_and_dataset_config_resolve_inside_project():
    assert ROOT_DIR.name == "project01_rebuild"
    assert CONFIG_PATH.is_file()
