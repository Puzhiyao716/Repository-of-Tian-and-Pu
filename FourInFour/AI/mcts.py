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
    EMPTY, TOTAL_CELLS, pos_4d_to_index, pos_2d_to_4d, pos_index_to_4d, pos_4d_to_2d, check_win_at,
)

# UCB1 探索参数（√2 是理论最优值）
UCB_C = 1.414


# ============================================================================
# MCTS 节点
# ============================================================================

class _Node:
    """MCTS 搜索树节点。"""

    __slots__ = ("board", "turn", "move", "parent", "children", "visits", "wins", "_untried", "winner")

    def __init__(self, board, turn, move=None, parent=None):
        self.board = board               # 棋盘状态（256 长度列表）
        self.turn = turn                 # 当前回合玩家
        self.move = move                 # 到达此节点的落子 (row, col)
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = {1: 0, 2: 0, 3: 0}
        self._untried = None
        self.winner = None
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

    def __init__(self, player_id, time_limit=5.0, max_iters=20000):
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
            #print("ITER", _, "root visits", root.visits, "children", len(root.children), "untried", len(root.get_untried_moves()))

        if not root.children:
            raise RuntimeError("无可落子位置")

        best = max(root.children, key=lambda c: c.visits)
        #选择一个次优解
        second_best = sorted(root.children, key=lambda c: c.visits, reverse=True)[1]
        #for c in root.children:
        #    print(c.move, c.visits, c.wins)
        print("Best move:", best.move, "Visits:", best.visits, "Wins:", best.wins)
        print("Second best move:", second_best.move, "Visits:", second_best.visits, "Wins:", second_best.wins)
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
            # ===== 新增：扩展前检查必胜一步 =====
            win_move = self._find_winning_move(node.board, node.turn)
            if win_move is not None:
                return self._expand_specific(node, win_move)
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

    def _expand_specific(self, parent, idx):
        """扩展一个指定落子为子节点（必胜一步）。"""
        w, x, y, z = pos_index_to_4d(idx)
        row, col = pos_4d_to_2d(w, x, y, z)

        new_board = list(parent.board)
        new_board[idx] = parent.turn
        next_turn = (parent.turn % 3) + 1

        child = _Node(new_board, next_turn, move=(row, col), parent=parent)
        
        # 这个一定赢
        child.winner = parent.turn
        parent.children.append(child)
        return child

    # ---- Simulation ----

    @staticmethod
    def _simulate(node):
        """带必胜判断的模拟至终局，返回胜者编号（0=平局）。"""

        # 如果这个节点已经结束
        if node.winner is not None:
            return node.winner
        board = list(node.board)
        turn = node.turn

        for _ in range(TOTAL_CELLS - sum(1 for v in board if v != EMPTY)):
            if _ < 2:
                # 前两步下，当前玩家若有一步获胜，立即执行
                win_move = MCTSEngine._find_winning_move(board, turn)
                if win_move is not None:
                    board[win_move] = turn
                    return turn

            # 无必胜点，随机选择一步
            empty = [i for i, v in enumerate(board) if v == EMPTY]
            if not empty:
                return 0
            idx = random.choice(empty)
            board[idx] = turn
            if check_win_at(board, idx, turn):
                return turn
            turn = (turn % 3) + 1

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
        # 已经有人赢
        if node.winner is not None:
            return True
        # 棋盘满
        if not any(v == EMPTY for v in node.board):
            node.winner = 0
            return True
        return False

    @staticmethod
    def _find_winning_move(board, player):
        """
        查找 player 是否存在一步获胜的落子。
        存在则返回该位置索引，否则返回 None。
        """
        for idx, v in enumerate(board):
            if v == EMPTY:
                board[idx] = player

                if check_win_at(board, idx, player):
                    board[idx] = EMPTY
                    return idx

                board[idx] = EMPTY

        return None
