"""最小费用最大流（Successive Shortest Path + 势函数 Dijkstra）。

用途：
1. 独占货 -> 车型的二分匹配费用下界（每票独占货必须单独成车）。
2. 给定方案每辆车的货物组合后，求各车的最优车型选择（容量可行、可用量受限）。

复杂度 O(F * E log V)，本场景下 F、V 均为几百量级。
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

INF = float("inf")


@dataclass
class _Edge:
    to: int
    rev: int
    cap: int
    cost: float
    initial_cap: int = 0


@dataclass
class FlowResult:
    feasible: bool
    total_cost: float
    # 每条 add_edge 对应的实际流量（按 add 顺序）。
    flows: list[int]


@dataclass
class MinCostFlow:
    n: int
    _g: list[list[_Edge]] = field(default_factory=list)
    _edge_index: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._g = [[] for _ in range(self.n)]

    def add_edge(self, fr: int, to: int, cap: int, cost: float) -> int:
        idx = len(self._edge_index)
        forward = _Edge(to, len(self._g[to]), cap, cost, initial_cap=cap)
        backward = _Edge(fr, len(self._g[fr]), 0, -cost)
        self._g[fr].append(forward)
        self._g[to].append(backward)
        self._edge_index.append((fr, len(self._g[fr]) - 1))
        return idx

    def solve(self, s: int, t: int, required_flow: int) -> FlowResult:
        n = self.n
        potential = [0.0] * n
        flow = 0
        total_cost = 0.0

        while flow < required_flow:
            dist = [INF] * n
            prev_v = [-1] * n
            prev_e = [-1] * n
            dist[s] = 0.0
            pq: list[tuple[float, int]] = [(0.0, s)]
            while pq:
                d, v = heapq.heappop(pq)
                if d != dist[v]:
                    continue
                for ei, e in enumerate(self._g[v]):
                    if e.cap <= 0:
                        continue
                    nd = d + e.cost + potential[v] - potential[e.to]
                    if nd + 1e-12 < dist[e.to]:
                        dist[e.to] = nd
                        prev_v[e.to] = v
                        prev_e[e.to] = ei
                        heapq.heappush(pq, (nd, e.to))
            if dist[t] == INF:
                return FlowResult(False, 0.0, [])
            for v in range(n):
                if dist[v] < INF:
                    potential[v] += dist[v]
            add = required_flow - flow
            v = t
            while v != s:
                add = min(add, self._g[prev_v[v]][prev_e[v]].cap)
                v = prev_v[v]
            flow += add
            v = t
            while v != s:
                pe, ei = prev_v[v], prev_e[v]
                e = self._g[pe][ei]
                total_cost += add * e.cost
                e.cap -= add
                self._g[v][e.rev].cap += add
                v = pe

        flows = []
        for fr, ei in self._edge_index:
            e = self._g[fr][ei]
            flows.append(e.initial_cap - e.cap)
        return FlowResult(True, total_cost, flows)
