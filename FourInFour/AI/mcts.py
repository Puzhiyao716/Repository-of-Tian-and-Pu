"""
稳健型 AI —— 蒙特卡洛树搜索 (MCTS) + UCB1
========================================

核心思想：
    1. 以自己赢为第一目的
    2. 假设所有对手也各自为赢而奋斗
    3. 用大量随机模拟来评估每个落子的胜率
    4. UCB1 公式在「利用已知好着」与「探索未知着」之间平衡

算法流程：
    Selection   — 从根出发，每层由当前回合玩家选 UCB1 最高的子节点
    Expansion   — 到叶子后随机扩展一个未尝试的落子
    Simulation  — 从新节点随机模拟至终局，记录胜者
    Backprop    — 结果沿路径向上传播
"""

import math
import random
import time
from typing import List, Tuple, Optional, Dict

from FourInFour.GameCore import (
    EMPTY, TOTAL_CELLS, pos_4d_to_index, pos_2d_to_4d, check_win_at,
)

# UCB1 探索参数（√2 是理论最优值）
UCB_C = 1.414


# ============================================================================
# MCTS 节点
# ============================================================================

class _Node:
    """MCTS 搜索树节点。"""

    __slots__ = ("board", "turn", "move", "move_idx", "parent", "children",
                 "visits", "wins", "_untried", "winner", "key_points")

    def __init__(
        self,
        board: List[int],
        turn: int,
        move: Optional[Tuple[int, int]] = None,
        parent: Optional["_Node"] = None,
        move_idx: int = -1,
        key_points: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self.board: List[int] = board            # 棋盘快照（256 长度列表）
        self.turn: int = turn                    # 当前回合玩家编号（1/2/3）
        self.move: Optional[Tuple[int, int]] = move  # 到达此节点的落子 (row, col)
        self.move_idx: int = move_idx            # 到达此节点的落子一维索引（root 为 -1）
        self.parent: Optional["_Node"] = parent  # 父节点（根节点为 None）
        self.children: List["_Node"] = []        # 已扩展的子节点列表
        self.visits: int = 0                     # 被访问次数（用于 UCB1 计算）
        self.wins: Dict[int, int] = {1: 0, 2: 0, 3: 0}  # 以各玩家视角的获胜次数
        self._untried: Optional[List[Tuple[int, int]]] = None  # 尚未尝试的落子（惰性初始化）
        self.winner: Optional[int] = None        # 必胜/必败标记（0=平局, None=未确定）
        self.key_points: List[Tuple[int, int]] = key_points or []  # root 节点优先探索的必胜点


    def get_untried_moves(self) -> List[Tuple[int, int]]:
        """返回当前棋盘上所有空位。root 节点的 Key_Point 优先，其余打乱。"""
        if self._untried is None:
            all_moves: List[Tuple[int, int]] = []
            for i, v in enumerate(self.board):
                if v == EMPTY:
                    _w = i // 64; r = i % 64
                    _x = r // 16; r = r % 16
                    _y = r // 4; _z = r % 4
                    all_moves.append((4 * _w + _x, 4 * _y + _z))

            if self.key_points:
                # Key_Point 放末尾（pop() 从末尾取，确保优先弹出）
                kp_set = {tuple(kp) for kp in self.key_points}
                priority = [m for m in self.key_points if tuple(m) in kp_set]
                remaining = [m for m in all_moves if tuple(m) not in kp_set]
                random.shuffle(remaining)
                self._untried = remaining + priority
            else:
                random.shuffle(all_moves)
                self._untried = all_moves
        return self._untried

    def ucb1(self, player: int) -> float:
        """本节点在指定玩家视角下的 UCB1 值。未访问过 → 无穷大。"""
        if self.visits == 0:
            return float("inf")
        return (self.wins[player] / self.visits) + UCB_C * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, player: int) -> "_Node":
        """返回该玩家视角下 UCB1 最高的子节点。"""
        return max(self.children, key=lambda c: c.ucb1(player))


# ============================================================================
# MCTS 搜索引擎
# ============================================================================

class MCTSEngine:
    """
    蒙特卡洛树搜索引擎。

    player_id : AI 自己的玩家编号（1/2/3）
    time_limit: 每次思考时间上限（秒）
    max_iters : 每次搜索迭代上限
    """

    def __init__(self, player_id: int, time_limit: float = 5.0, max_iters: int = 10000) -> None:
        self.player_id: int = player_id     # AI 自己的玩家编号（1/2/3）
        self.time_limit: float = time_limit # 每次思考时间上限（秒）
        self.max_iters: int = max_iters     # 每次搜索迭代次数上限

    def choose_move(
        self, board: List[int], turn: int,
        last_moves: Dict[int, int] = None
    ) -> Tuple[Tuple[int, int], float, int]:
        """
        给定棋盘和当前回合，返回 AI 选择的落子与思考统计。

        last_moves: {player_id: 一维索引}，所有玩家最近一步棋。
                    用于 root 节点计算 Key_Point（必胜点优先探索）。

        返回: ((row, col), elapsed_time_seconds, iterations_count)
        """
        # ---- 计算 root 节点的 Key_Point ----
        key_points = self._compute_key_points(board, last_moves or {})

        root = _Node(list(board), turn, key_points=key_points)

        t0 = time.time()
        actual_iters = 0
        for _ in range(self.max_iters):
            if time.time() - t0 > self.time_limit:
                break

            leaf = self._select(root)               # 1. Selection + Expansion
            winner = self._simulate(leaf)            # 2. Simulation
            self._backprop(leaf, winner)             # 3. Backpropagation
            actual_iters += 1

        if not root.children:
            raise RuntimeError("无可落子位置")

        best = max(root.children, key=lambda c: c.visits)
        elapsed = time.time() - t0
        print("Best move:", best.move, "Visits:", best.visits, "Wins:", best.wins,
              "Time:", f"{elapsed:.3f}s", "Iters:", actual_iters)
        # 选择一个次优解
        second_best = sorted(root.children, key=lambda c: c.visits, reverse=True)[1]
        print("Second best move:", second_best.move, "Visits:", second_best.visits, "Wins:", second_best.wins)
        return best.move, elapsed, actual_iters

    # ---- Selection & Expansion ----

    def _select(self, node: _Node) -> _Node:
        """
        从根出发选叶节点。每层由「当前回合玩家」选对自己最有利的子节点，
        体现「每个对手都为自己的胜利而奋斗」。

        到达叶节点后，从 untried 列表 pop 一个落子扩展（Key_Point 优先）。
        """
        while True:
            if self._is_terminal(node):
                return node

            untried = node.get_untried_moves()
            if untried:
                return self._expand(node, untried)

            node = node.best_child(node.turn)

    def _expand(
        self, parent: _Node, untried: List[Tuple[int, int]]
    ) -> _Node:
        """扩展一个未尝试落子为子节点。"""
        row, col = untried.pop()
        w, x, y, z = pos_2d_to_4d(row, col)
        idx = pos_4d_to_index(w, x, y, z)

        new_board = list(parent.board)
        new_board[idx] = parent.turn
        next_turn = (parent.turn % 3) + 1

        child = _Node(new_board, next_turn, move=(row, col),
                      parent=parent, move_idx=idx)
        parent.children.append(child)
        return child

    # ---- Simulation ----

    def _simulate(self, node: _Node) -> int:
        """
        带必胜队列引导的模拟至终局，返回胜者编号（0=平局）。

        策略：维护一个 FIFO 必胜队列 key_queue。
              每步落子后检测当前玩家是否形成必胜点，
              若有则追加到队列末尾；下一玩家优先从队列头部取空位落子。
        """
        # 如果这个节点已经结束
        if node.winner is not None:
            return node.winner

        board = list(node.board)
        turn = node.turn
        # FIFO 队列：存储一维索引，优先落子位置
        key_queue: List[int] = []

        remaining = TOTAL_CELLS - sum(1 for v in board if v != EMPTY)
        for _ in range(remaining):
            # ---- 选择落子：优先从队列头部取 ----
            idx: int = -1
            while key_queue:
                candidate = key_queue.pop(0)
                if board[candidate] == EMPTY:
                    idx = candidate
                    break

            if idx == -1:
                # 队列无有效位置，随机选择
                empty = [i for i, v in enumerate(board) if v == EMPTY]
                if not empty:
                    return 0
                idx = random.choice(empty)

            # ---- 落子 ----
            board[idx] = turn

            # ---- 检测：当前玩家是否胜利？是否有必胜点？----
            won, pot_points = check_win_at(board, idx, turn, potential_win=True)
            if won:
                return turn

            # 将必胜点的一维索引追加到队列末尾
            if pot_points:
                for pot_row, pot_col in pot_points:
                    pot_w, pot_x = pot_row // 4, pot_row % 4
                    pot_y, pot_z = pot_col // 4, pot_col % 4
                    pot_idx = pos_4d_to_index(pot_w, pot_x, pot_y, pot_z)
                    if board[pot_idx] == EMPTY:
                        key_queue.append(pot_idx)

            # ---- 轮换回合 ----
            turn = (turn % 3) + 1

        return 0

    # ---- Backpropagation ----

    @staticmethod
    def _backprop(node: _Node, winner: int) -> None:
        """结果向上传播。"""
        while node is not None:
            node.visits += 1
            if winner != 0:
                node.wins[winner] += 1
            node = node.parent

    # ---- 辅助 ----

    def _compute_key_points(
        self, board: List[int], last_moves: Dict[int, int]
    ) -> List[Tuple[int, int]]:
        """
        root 节点专用：根据最近各玩家落子位置，收集必胜点。

        对每个玩家的最后落子位置，用 check_win_at 按 我→下家→上家 顺序检测，
        收集所有三子一空的潜在获胜位置，去重后返回。

        参数：
            board:      当前棋盘
            last_moves: {player_id: 一维索引}，各玩家最近一步棋
        返回：
            去重后的必胜点列表 [[row, col], ...]
        """
        if not last_moves:
            return []

        ai = self.player_id
        next_p = (ai % 3) + 1          # 下家
        prev_p = ((ai - 2) % 3) + 1    # 上家
        seen: set = set()
        result: List[Tuple[int, int]] = []

        for _pid, idx in last_moves.items():
            for check_p in (ai, next_p, prev_p):
                _won, pot_points = check_win_at(board, idx, check_p,
                                                potential_win=True)
                if pot_points:
                    for pt in pot_points:
                        key = (pt[0], pt[1])
                        if key not in seen:
                            seen.add(key)
                            result.append(key)
        return result

    @staticmethod
    def _is_terminal(node: _Node) -> bool:
        # 已经有人赢
        if node.winner is not None:
            return True
        # 棋盘满
        if not any(v == EMPTY for v in node.board):
            node.winner = 0
            return True
        return False
