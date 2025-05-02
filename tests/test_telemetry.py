from pathlib import Path

from cloudai import Registry, Telemetry


def test_one():
    t = Telemetry()
    r = Registry()
    assert isinstance(t, Telemetry)
    assert not isinstance(t, Registry)
    assert isinstance(r, Registry)
    assert not isinstance(r, Telemetry)

    t.set_output_dir(Path.cwd())
    t.set_name("test")


def test_same():
    t1 = Telemetry()
    t2 = Telemetry()
    assert t1 is t2

    r1 = Registry()
    r2 = Registry()
    assert r1 is r2
