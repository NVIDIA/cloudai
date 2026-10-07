# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Callable
from unittest.mock import Mock, patch

import pytest
import toml

from cloudai.test_parser import TestParser


class TestParseAllFailureCollection:
    @pytest.fixture()
    def config_dir(self, tmp_path: Path, write_parseable_test: Callable[[Path, str], None]) -> Path:
        config_dir = tmp_path / "tests"
        config_dir.mkdir()
        write_parseable_test(config_dir / "good_test.toml", "good_test")
        return config_dir

    def test_parse_all_keeps_parseable_configs_and_returns_each_failure_by_path(
        self, config_dir: Path, write_unregistered_agent_test: Callable[[Path, str], None]
    ):
        unregistered_agent_toml = config_dir / "bad_agent_test.toml"
        write_unregistered_agent_test(unregistered_agent_toml, "bad_agent_test")
        malformed_toml = config_dir / "broken_toml_test.toml"
        malformed_toml.write_text('name = "broken_toml_test"\ndescription = \n')

        test_parser = TestParser(sorted(config_dir.glob("*.toml")), None)  # type: ignore

        parsed, failures = test_parser.parse_all()

        assert [test.name for test in parsed] == ["good_test"]
        assert set(failures) == {unregistered_agent_toml, malformed_toml}
        assert "is not registered" in str(failures[unregistered_agent_toml].error)
        assert "TOML parsing error" in str(failures[malformed_toml].error)
        assert failures[unregistered_agent_toml].declared_name == "bad_agent_test"
        assert failures[malformed_toml].declared_name is None

    def test_parse_all_still_raises_on_duplicate_name_across_files(
        self, config_dir: Path, write_parseable_test: Callable[[Path, str], None]
    ):
        duplicate = config_dir / "good_test_copy.toml"
        write_parseable_test(duplicate, "good_test")

        test_parser = TestParser(sorted(config_dir.glob("*.toml")), None)  # type: ignore

        with pytest.raises(ValueError, match="good_test") as exc_info:
            test_parser.parse_all()

        msg = str(exc_info.value)
        assert str(duplicate) in msg
        assert str(config_dir / "good_test.toml") in msg


class TestParseAllDuplicateDetection:
    @staticmethod
    def _write_test_toml(path: Path, name: str) -> None:
        path.write_text(toml.dumps({"name": name, "description": "test"}))

    @staticmethod
    def _make_fake_parsed(name: str) -> Mock:
        obj = Mock()
        obj.name = name
        return obj

    def test_duplicate_name_across_files_raises(self, tmp_path: Path):
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        dse_toml = test_dir / "dse_qwen_30b_a3b.toml"
        single_toml = test_dir / "qwen_30b_a3b.toml"
        self._write_test_toml(dse_toml, "dse_qwen_30b_a3b")
        self._write_test_toml(single_toml, "dse_qwen_30b_a3b")

        parser = TestParser([dse_toml, single_toml], None)  # type: ignore
        fake = self._make_fake_parsed("dse_qwen_30b_a3b")

        with patch.object(parser, "_parse_data", return_value=fake):
            with pytest.raises(ValueError, match="dse_qwen_30b_a3b") as exc_info:
                parser.parse_all()
            msg = str(exc_info.value)
            assert str(dse_toml) in msg
            assert str(single_toml) in msg

    def test_unique_names_no_error(self, tmp_path: Path):
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        toml_a = test_dir / "test_a.toml"
        toml_b = test_dir / "test_b.toml"
        self._write_test_toml(toml_a, "test_a")
        self._write_test_toml(toml_b, "test_b")

        parser = TestParser([toml_a, toml_b], None)  # type: ignore

        with patch.object(
            parser,
            "_parse_data",
            side_effect=[self._make_fake_parsed("test_a"), self._make_fake_parsed("test_b")],
        ):
            results, failures = parser.parse_all()
            assert len(results) == 2
            assert failures == {}

    def test_single_file_no_error(self, tmp_path: Path):
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        toml_a = test_dir / "only_one.toml"
        self._write_test_toml(toml_a, "only_one")

        parser = TestParser([toml_a], None)  # type: ignore

        with patch.object(parser, "_parse_data", return_value=self._make_fake_parsed("only_one")):
            results, failures = parser.parse_all()
            assert len(results) == 1
            assert failures == {}

    def test_duplicate_toml_key_reports_file_and_key(self, tmp_path: Path):
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        broken_toml = test_dir / "broken.toml"
        broken_toml.write_text('name = "test_a"\nname = "test_b"\n')

        parser = TestParser([broken_toml], None)  # type: ignore

        parsed, failures = parser.parse_all()

        assert parsed == []
        msg = str(failures[broken_toml])
        assert "duplicate TOML key 'name'" in msg
        assert str(broken_toml) in msg
        assert "line 2, column 1" in msg
