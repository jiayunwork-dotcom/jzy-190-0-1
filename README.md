# 城配排车服务

异构车型、类别互斥、独占约束的 **anytime 排车作业服务**。调度系统只经 HTTP 调用。

- Python 3.12 / FastAPI / SQLite（数据文件挂卷、WAL 模式）
- 提交即返回作业号；后台持续改进；随时查当前最好方案、下界、相对差距；可取消
- 每次都给**费用下界**并报告是 重量/体积/独占/互斥/平凡 中哪一维起决定作用
- 方案费用触达下界即标记"已证明最优"，否则超时/取消都交出完整可行方案
- 同输入同种子结果逐位一致；重启后已完成作业保留、未完成作业标记为中断
- 算法细节、适用数据局限、时间增长见 [`docs/算法说明.md`](docs/算法说明.md)

## 运行

### 容器

```bash
docker build -t dispatch:latest .
docker run -d -p 8000:8000 -v $(pwd)/data:/data dispatch:latest
```

SQLite 文件默认在容器内 `/data/dispatch.db`（`DISPATCH_DB_PATH` 可改），挂卷持久化。

### 本地

```bash
pip install -r requirements-dev.txt
pytest                       # 单元测试
uvicorn app.main:app --port 8000
```

## 接口

### 1) 提交作业 `POST /jobs` → 202

```json
{
  "shipments": [
    {"id": "s1", "weight": 6, "volume": 0.2},
    {"id": "s2", "weight": 5, "volume": 0.2, "category": "food"},
    {"id": "s3", "weight": 5, "volume": 0.2, "category": "chem"},
    {"id": "s4", "weight": 4, "volume": 0.2, "exclusive": true}
  ],
  "vehicle_types": [
    {"id": "T10", "max_weight": 10, "max_volume": 60, "cost": 100, "available": 8},
    {"id": "T20", "max_weight": 20, "max_volume": 120, "cost": 180, "available": 4}
  ],
  "conflicts": [["food", "chem"], ["chem", "chem"]],
  "seed": 42,
  "time_limit_seconds": 30
}
```

- `weight`（吨）、`volume`（方）非负、有限；`cost` 非负；`available` 为非负整数。
- `category` 可空；`conflicts` 为无序类别对，`["X","X"]` 表示 X 类自冲突。
- 立即返回 `{"job_id": "...", "status": "pending"}`。

### 2) 查询 `GET /jobs/{job_id}`

返回状态、迭代数、当前最好费用/车辆数、下界各维明细、`relative_gap`（=(费用−下界)/下界）、
差距百分比，以及完整方案（每车车型、货号、重量、体积）。终态方案读回时会再次独立校验。

关键字段：

- `status`：`pending/running/completed/timeout/cancelled/interrupted/infeasible`
- `stop_reason`：`proven_optimal / time_limit / cancelled / interrupted / infeasible / running / pending`
- `lower_bound.binding`：起决定作用的下界维度

### 3) 取消 `POST /jobs/{job_id}/cancel`

取消在求解的安全检查点生效，终态方案必须完整可行。

### 4) 健康 `GET /healthz`

## 校验错误

类型/格式错误（负数、NaN、缺字段等）与业务错误（货比任何车型都大、独占货数超过车型总数、
冲突引用不存在的类别、货号/车型号重复等）都返回 **422**，错误体逐字段给出 `field` 与 `message`：

```json
{"error": "validation_failed",
 "detail": [{"field": "shipments[3].weight", "message": "weight 不能为负数"}]}
```

## 行为约定（已由测试覆盖）

- 6/5/5/4 吨、单车型 10 吨：下界 2 辆，给出 2 辆方案并证明最优；
- 全部货装得进一辆车时只用一辆；
- 每个方案满足恰好覆盖/不超载/不超容/冲突不同车/独占独占/车型用量上限；
- 方案费用 ≥ 下界；多加一票货下界不降；重量与载重同乘正数车辆数不变；
- 同输入同种子结果一致；取消与超时都交出完整可行方案；重启后中断标记与方案保留。
