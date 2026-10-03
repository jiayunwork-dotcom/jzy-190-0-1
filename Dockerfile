FROM python:3.12-slim

# 不生成 .pyc、不缓冲 stdout/stderr，便于容器日志即时可见。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DISPATCH_DB_PATH=/data/dispatch.db

WORKDIR /srv

# 先装依赖以利用层缓存。
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷业务代码。
COPY app ./app

# SQLite 数据文件所在目录，运行时挂卷到宿主机。
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# 单进程 + 线程池执行作业；如横向扩展可改为多副本（各自独立 SQLite 文件）。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
