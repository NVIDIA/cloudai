from cloudai.cli.handlers import is_dse_job

# Mock data for testing
mock_toml_dse = {
    "test": {
        "cmd_args": {
            "docker_image_url": "https://docker/fake_url",
            "load_container": True,
            "FakeConfig": {
                "policy": ["option1", "option2"],
                "shape": "[1, 2, 3, 4]",
                "dtype": "fake_type",
                "mesh_shape": "[4, 3, 2, 1]",
            },
        }
    }
}

mock_toml_non_dse = {
    "test": {
        "cmd_args": {
            "docker_image_url": "https://docker/fake_url",
            "load_container": True,
            "FakeConfig": {
                "policy": "option1",
                "shape": "[1, 2, 3, 4]",
                "dtype": "fake_type",
            },
        }
    }
}


def test_is_dse_job_dse():
    assert is_dse_job(mock_toml_dse["test"]["cmd_args"])


def test_is_dse_job_non_dse():
    assert not is_dse_job(mock_toml_non_dse["test"]["cmd_args"])
