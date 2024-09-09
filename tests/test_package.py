import toml


def test_requirements():
    """
    Test that the requirements in the requirements.txt file are the same as the requirements in the pyproject.toml file.
    """
    with open("requirements.txt", "r") as f:
        requirements_txt = sorted(f.read().splitlines())
    with open("pyproject.toml", "r") as f:
        requirements_toml = sorted(toml.load(f)["project"]["dependencies"])
    assert requirements_txt == requirements_toml
