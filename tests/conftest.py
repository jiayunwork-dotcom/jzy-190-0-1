import os
import tempfile
import time

# 导入 app 之前把默认数据文件指到临时目录（模块级 app 会在导入时创建）
os.environ.setdefault("APP_DB_PATH", os.path.join(tempfile.mkdtemp(), "default.db"))

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def make_problem(items, vehicle_types, categories=None, conflicts=None):
    return {
        "items": items,
        "vehicle_types": vehicle_types,
        "categories": categories or [],
        "conflicts": conflicts or [],
    }


def submit(client, problem, seed=0, time_limit=30.0):
    r = client.post(
        "/jobs",
        json={"problem": problem, "options": {"seed": seed, "time_limit_seconds": time_limit}},
    )
    assert r.status_code == 201, r.json()
    return r.json()


def wait_done(client, job_id, timeout=60.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("completed", "cancelled", "interrupted", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")
