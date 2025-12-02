from pathlib import Path
from unittest.mock import patch

import pytest

from cloudai.core import HFModel
from cloudai.util.hf_model_manager import HFModelManager


@pytest.fixture
def hf_model() -> HFModel:
    return HFModel(model_name="some_model_name")


def test_download(hf_model: HFModel, tmp_path: Path) -> None:
    assert hf_model._installed_path is None

    with patch("cloudai.util.hf_model_manager.snapshot_download", return_value=str("/real/path")):
        HFModelManager(root_path=tmp_path).download_model(hf_model)

    assert hf_model.installed_path == Path("/real/path")


def test_download_raises(hf_model: HFModel, tmp_path: Path) -> None:
    with patch(
        "cloudai.util.hf_model_manager.snapshot_download",
        side_effect=Exception("some error message"),
    ):
        result = HFModelManager(root_path=tmp_path).download_model(hf_model)

    assert not result.success
    assert "some error message" in result.message
    assert hf_model._installed_path is None
