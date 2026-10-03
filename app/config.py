"""运行配置：全部来自环境变量，容器里只挂卷、只走 HTTP。"""
from __future__ import annotations

import os

DEFAULT_DB_PATH = os.environ.get("DISPATCH_DB_PATH", "/data/dispatch.db")
# 进度落盘的最小间隔（秒）：保证"随时可查"又不把 SQLite 打爆。
PROGRESS_FLUSH_INTERVAL = float(os.environ.get("DISPATCH_FLUSH_INTERVAL", "0.2"))
# 取消信号检查到线程退出的时间上限只是经验值，不影响正确性。
