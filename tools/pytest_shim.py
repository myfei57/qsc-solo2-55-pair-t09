#!/usr/bin/env python3
"""无 pytest 环境下的最小兼容垫片：收集 tests/ 下全部用例并执行。

支持本仓库实际用到的 pytest 子集：模块级函数、``pytest.raises``、
``pytest.approx``、``tmp_path`` 与 conftest 里的 ``clock/platform/online``
夹具。正式门禁仍以 pytest 为准（见 requirements.txt）。
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import pathlib
import sys
import tempfile
import traceback
from contextlib import contextmanager

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cspplant.app import build_plant  # noqa: E402
from cspplant.clock import ManualClock  # noqa: E402
from cspplant.config import default_config  # noqa: E402


class _Approx:
    def __init__(self, expected, rel=1e-6, abs_=1e-9):
        self.expected = expected
        self.rel = rel
        self.abs = abs_

    def __eq__(self, other):
        return abs(other - self.expected) <= max(self.rel * abs(self.expected), self.abs)


class _Raises:
    def __init__(self, exc_type):
        self.exc_type = exc_type
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.exc_type.__name__}")
        if not issubclass(exc_type, self.exc_type):
            return False
        self.value = exc
        return True


class _Pytest:
    @staticmethod
    def approx(expected, rel=1e-6, abs=1e-9):
        return _Approx(expected, rel, abs)

    @staticmethod
    def raises(exc_type):
        return _Raises(exc_type)

    @staticmethod
    def skip(reason=""):
        raise _Skipped(reason)


class _Skipped(Exception):
    pass


sys.modules.setdefault("pytest", _Pytest)


class _TmpPath:
    """tmp_path 垫片：拼目录自动建目录，拼文件只建父目录，其余委托给 Path。"""

    def __init__(self, path):
        self._path = pathlib.Path(path)
        self._path.mkdir(parents=True, exist_ok=True)

    def __truediv__(self, other):
        child = self._path / other
        if not child.suffix:
            child.mkdir(parents=True, exist_ok=True)
            return _TmpPath(child)
        child.parent.mkdir(parents=True, exist_ok=True)
        return child

    def __fspath__(self):
        return str(self._path)

    def __str__(self):
        return str(self._path)

    def __getattr__(self, name):
        return getattr(self._path, name)


class _Capture:
    """capsys 垫片：readouterr 读出并清空；close 在用例结束后恢复标准流。"""

    def __init__(self):
        import io
        import contextlib

        self._out = io.StringIO()
        self._err = io.StringIO()
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(contextlib.redirect_stdout(self._out))
        self._stack.enter_context(contextlib.redirect_stderr(self._err))

    def readouterr(self):
        class _Result:
            def __init__(self, out, err):
                self.out = out
                self.err = err

        result = _Result(self._out.getvalue(), self._err.getvalue())
        self._out.seek(0)
        self._out.truncate(0)
        self._err.seek(0)
        self._err.truncate(0)
        return result

    def close(self):
        self._stack.close()


class _FixtureProvider:
    """函数级惰性夹具：只在用例实际请求时才构造，互不污染。"""

    def __init__(self):
        self._cache: dict = {}

    def get(self, name: str):
        if name in self._cache:
            return self._cache[name]
        if name == "tmp_path":
            value = _TmpPath(tempfile.mkdtemp(prefix="pyshim-"))
        elif name == "clock":
            value = ManualClock()
        elif name == "platform":
            value = build_plant(
                default_config(),
                clock=self.get("clock"),
                storage_root=str(self.get("tmp_path") / "state"),
            )
        elif name == "online":
            value = _build_online(self.get("platform"))
        elif name == "capsys":
            value = _Capture()
        else:
            raise TypeError(f"unknown fixture {name!r}")
        self._cache[name] = value
        return value


def _bind_kwargs(fn):
    sig = inspect.signature(fn)
    provider = _FixtureProvider()
    return {pname: provider.get(pname) for pname in sig.parameters}


def _build_online(platform):
    platform.start()
    platform.track(800.0, 720.0)
    platform.circulate()
    platform.dispatch(100.0, "fixture")
    return platform


def main():
    tests_dir = ROOT / "tests"
    files = sorted(tests_dir.glob("test_*.py"))
    passed = failed = skipped = 0
    failures = []

    for file in files:
        mod_name = f"shim_{file.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, file)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            failed += 1
            failures.append((f"{file.name}::<import>", traceback.format_exc()))
            continue

        test_fns = [
            (name, obj)
            for name, obj in vars(module).items()
            if name.startswith("test_") and callable(obj)
        ]
        for name, fn in test_fns:
            kwargs = _bind_kwargs(fn)
            capture = kwargs.get("capsys")
            try:
                fn(**kwargs)
                passed += 1
                print(".", end="", flush=True)
            except _Skipped as exc:
                skipped += 1
                print("s", end="", flush=True)
            except Exception:
                failed += 1
                failures.append((f"{file.name}::{name}", traceback.format_exc()))
                print("F", end="", flush=True)
            finally:
                if capture is not None:
                    capture.close()

    print()
    for label, tb in failures:
        print("=" * 70)
        print("FAIL", label)
        print(tb)
    print(f"{passed} passed, {failed} failed, {skipped} skipped, {len(list(files))} files")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
