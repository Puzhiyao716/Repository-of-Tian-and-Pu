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
from typing import List, Tuple, Optional

from FourInFour.GameCore.board import (
    EMPTY, TOTAL_CELLS, pos_4d_to_index, pos_2d_to_4d, check_win_at,
)

# UCB1 探索参数（√2 是理论最优值）
UCB_C = 1.414


# ============================================================================
# MCTS 节点
# ============================================================================

class _Node:
    """MCTS 搜索树节点。"""

    __slots__ = ("board", "turn", "move", "parent", "children", "visits", "wins", "_untried")

    def __init__(self, board, turn, move=None, parent=None):
        self.board = board               # 棋盘状态（256 长度列表）
        self.turn = turn                 # 当前回合玩家
        self.move = move                 # 到达此节点的落子 (row, col)
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = {1: 0, 2: 0, 3: 0}
        self._untried = None

    def get_untried_moves(self):
        """返回当前棋盘上所有空位（打乱以增加多样性）。"""
        if self._untried is None:
            moves = []
            for i, v in enumerate(self.board):
                if v == EMPTY:
                    _w = i // 64; r = i % 64
                    _x = r // 16; r = r % 16
                    _y = r // 4; _z = r % 4
                    moves.append((4 * _w + _x, 4 * _y + _z))
            random.shuffle(moves)
            self._untried = moves
        return self._untried

    def ucb1(self, player):
        """本节点在指定玩家视角下的 UCB1 值。未访问过 → 无穷大。"""
        if self.visits == 0:
            return float("inf")
        return (self.wins[player] / self.visits) + UCB_C * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, player):
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

    def __init__(self, player_id, time_limit=2.0, max_iters=20000):
        self.player_id = player_id
        self.time_limit = time_limit
        self.max_iters = max_iters

    def choose_move(self, board, turn):
        """
        给定棋盘和当前回合，返回 AI 选择的落子 (row, col)。
        """
        root = _Node(list(board), turn)

        t0 = time.time()
        for _ in range(self.max_iters):
            if time.time() - t0 > self.time_limit:
                break

            leaf = self._select(root)               # 1. Selection + Expansion
            winner = self._simulate(leaf)            # 2. Simulation
            self._backprop(leaf, winner)             # 3. Backpropagation

        if not root.children:
            raise RuntimeError("无可落子位置")

        best = max(root.children, key=lambda c: c.visits)
        return best.move

    # ---- Selection & Expansion ----

    def _select(self, node):
        """
        从根出发选叶节点。每层由「当前回合玩家」选对自己最有利的子节点，
        体现「每个对手都为自己的胜利而奋斗」。
        """
        while True:
            if self._is_terminal(node):
                return node
            untried = node.get_untried_moves()
            if untried:
                return self._expand(node, untried)
            node = node.best_child(node.turn)

    def _expand(self, parent, untried):
        """扩展一个未尝试落子为子节点。"""
        row, col = untried.pop()
        w, x, y, z = pos_2d_to_4d(row, col)
        idx = pos_4d_to_index(w, x, y, z)

        new_board = list(parent.board)
        new_board[idx] = parent.turn
        next_turn = (parent.turn % 3) + 1

        child = _Node(new_board, next_turn, move=(row, col), parent=parent)
        parent.children.append(child)
        return child

    # ---- Simulation ----

    @staticmethod
    def _simulate(node):
        """随机模拟至终局，返回胜者编号（0=平局）。"""
        board = list(node.board)
        turn = node.turn
        empty = [i for i, v in enumerate(board) if v == EMPTY]
        random.shuffle(empty)

        for idx in empty:
            board[idx] = turn
            if check_win_at(board, idx, turn):
                return turn
            turn = (turn % 3) + 1

        return 0

    # ---- Backpropagation ----

    @staticmethod
    def _backprop(node, winner):
        """结果向上传播。"""
        while node is not None:
            node.visits += 1
            if winner != 0:
                node.wins[winner] += 1
            node = node.parent

    # ---- 辅助 ----

    @staticmethod
    def _is_terminal(node):
        """无空位即终局。"""
        return not node.get_untried_moves()
