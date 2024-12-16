from cloudai import TestRun, TestScenario
from cloudai._core.cases_iter import SequentialCasesIter
from tests.conftest import create_autospec_dataclass


def test_empty():
    cases_iter = SequentialCasesIter(test_scenario=TestScenario(name="ts", test_runs=[]))
    assert not cases_iter.has_more_cases
    idx = 0
    for _ in cases_iter:
        idx += 1
    assert idx == 0


def test_cases_iter():
    test_runs = [create_autospec_dataclass(TestRun) for _ in range(3)]
    cases_iter = SequentialCasesIter(test_scenario=TestScenario(name="ts", test_runs=test_runs))  # type: ignore
    assert cases_iter.has_more_cases

    # first go, onle one test run is ready
    idx = 0
    for tr in cases_iter:
        assert tr == test_runs[idx]
        idx += 1
    assert idx == 1
    assert cases_iter.has_more_cases
    idx = 0
    # even new loop creating doesn't trigger new cases
    for _ in cases_iter:
        idx += 1
    assert idx == 0

    # second case runs only after first one is completed
    cases_iter.on_completed(test_runs[0], None)  # type: ignore
    assert cases_iter.has_more_cases
    idx = 0
    for tr in cases_iter:
        assert tr == test_runs[1]
        idx += 1
    assert idx == 1
    assert cases_iter.has_more_cases
    assert len(cases_iter.ready_for_run) == 0
    idx = 0
    for _ in cases_iter:
        idx += 1
    assert idx == 0  # no cases until second is completed

    # last iteration
    cases_iter.on_completed(test_runs[1], None)  # type: ignore
    assert cases_iter.has_more_cases
    idx = 0
    for tr in cases_iter:
        assert tr == test_runs[2]
        idx += 1
    assert idx == 1

    cases_iter.on_completed(test_runs[2], None)  # type: ignore
    assert not cases_iter.has_more_cases
