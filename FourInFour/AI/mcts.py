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

    __slots__ = ("board", "turn", "move", "parent", "children", "visits", "wins", "_untried", "winner", "winning_moves")

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
        self.winning_moves = [[], [], [], []]  # 每个玩家的必胜一步列表

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

    def update_winning_moves(self):
        """更新前一步导致的的必胜一步列表变化。 若直接赢则返回 True，否则返回 False。"""
        from FourInFour.GameCore.board import LINES_BY_CELL
        player = (self.turn + 1) % 3 + 1  # 上一回合玩家
        w, x, y, z = pos_2d_to_4d(self.move[0], self.move[1])
        idx = pos_4d_to_index(w, x, y, z)

        # 删除必胜列表中的上一步落子
        if idx in self.winning_moves[player]:
            return True
        for p in range(1, 4):
            if idx in self.winning_moves[p]:
                self.winning_moves[p].remove(idx)
        
        # 只检查与上一步落子相关的连线
        self.winning_moves[player] = []
        for line in LINES_BY_CELL[idx]:
            count = 0
            empty_idx = None
            for idx in line:
                cell = self.board[idx]
                if cell == player:
                    count += 1
                elif cell == EMPTY:
                    empty_idx = idx
                else:
                    # 这条线上有对手棋子，不可能形成四连
                    count = -1
                    break
            if count == 3 and empty_idx is not None:
                self.winning_moves[player].append(empty_idx)
        return False

    def get_priority_move(self):
        """
        按优先级返回一个落子索引（优先级1→2→3），若无可选则返回 None。
        依赖 self.turn（当前玩家）、self.winning_moves（列表，每个玩家对应一个列表）
        """
        current = self.turn
        next_p = (current % 3) + 1
        last_p = (current + 1) % 3 + 1 

        # 1. 当前玩家有必胜点
        if self.winning_moves[current]:
            return self.winning_moves[current][0]

        # 2. 下一个玩家有必胜点
        if self.winning_moves[next_p]:
            last_moves = self.winning_moves[last_p]
            # 尝试选择一个 next 必胜点，该点最好能破坏 last 的必胜点，至少使得走完后 last 的必胜点数量 < 2
            # 走 move 后 last 剩余的必胜点数量 = 原数量 - (move 是否在 last_moves 中)
            remaining = len(last_moves) - (1 if self.winning_moves[next_p][0] in last_moves else 0)
            if remaining < 2:
                return self.winning_moves[next_p][0]   # 符合条件，选择此点阻止 next
        # 所有 next 必胜点都会导致 last 剩余 ≥2，放弃优先级2

        # 3. 如果 last player 有至少两个必胜点，返回其中一个
        if len(self.winning_moves[last_p]) >= 2:
            return self.winning_moves[last_p][0]

        # 无符合条件的点
        return None
        

    def ucb1(self, player):
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
        root = _Node(list(board), turn)
        for player in range (1, 4):
            root.winning_moves[player] = self._find_winning_moves(board, player)
        # print("Winning moves:", root.winning_moves)

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
        #print("winning moves after search:", best.winning_moves)
        elapsed = time.time() - t0
        print("player", self.player_id, "Best move:", best.move, "Visits:", best.visits, "Wins:", best.wins,
              "Time:", f"{elapsed:.3f}s", "Iters:", actual_iters)
        return best.move, elapsed, actual_iters

    # ---- Selection & Expansion ----

    def _select(self, node: _Node) -> _Node:
        """
        从根出发选叶节点。每层由「当前回合玩家」选对自己最有利的子节点，
        体现「每个对手都为自己的胜利而奋斗」。

        到达叶节点后，从 untried 列表 pop 一个落子扩展（Key_Point 优先）。
        """
        for _ in range(5):
            if self._is_terminal(node):
                return node

            priority_move = node.get_priority_move()
            if priority_move is not None:
                # 有优先落子，直接返回该节点
                if node.children:
                    node = node.children[0]
                    if node.winner is not None:
                        return node
                else: 
                    node = self._expand_priority_move(node, priority_move)
                    if node.update_winning_moves():
                        # 此为必胜一步，当前玩家获胜
                        node.winner = node.parent.turn
                        return node

            else:
                # 无优先落子, 进入 UCB1 选择
                untried = node.get_untried_moves()
                if untried:
                    node = self._expand(node, untried)
                    if node.update_winning_moves():
                        # 此为必胜一步，当前玩家获胜
                        node.winner = node.parent.turn
                        return node
                    
                else:
                    node = node.best_child(node.turn)
                    if node.winner is not None:
                        return node
        return node

    def _expand(
        self, parent: _Node, untried: List[Tuple[int, int]]
    ) -> _Node:
        """扩展一个未尝试落子为子节点。"""
        row, col = untried.pop(0)
        w, x, y, z = pos_2d_to_4d(row, col)
        idx = pos_4d_to_index(w, x, y, z)

        new_board = list(parent.board)
        new_board[idx] = parent.turn
        next_turn = (parent.turn % 3) + 1

        child = _Node(new_board, next_turn, move=(row, col), parent=parent)
        child.winning_moves = [lst[:] for lst in parent.winning_moves]
            
        parent.children.append(child)
        return child

    def _expand_priority_move(self, parent, priority_move):
        """扩展优先落子为子节点。"""
        idx = priority_move
        w, x, y, z = pos_index_to_4d(idx)
        row, col = pos_4d_to_2d(w, x, y, z)

        new_board = list(parent.board)
        new_board[idx] = parent.turn
        next_turn = (parent.turn % 3) + 1

        child = _Node(new_board, next_turn, move=(row, col), parent=parent)
        child.winning_moves = [lst[:] for lst in parent.winning_moves]
            
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

        for _ in range(TOTAL_CELLS - sum(1 for v in board if v != EMPTY)):
            # 随机选择一步
            empty = [i for i, v in enumerate(board) if v == EMPTY]
            if not empty:
                return 0
            idx = random.choice(empty)
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
    def _is_terminal(node):
        # 棋盘满
        if not any(v == EMPTY for v in node.board):
            node.winner = 0
            return True
        return False

#    @staticmethod
#    def _find_winning_move(board, player):
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

    @staticmethod
    def _find_winning_moves(board, player):
        from FourInFour.GameCore.board import WINNING_LINES
        winning_moves = []
        for line in WINNING_LINES:
            count = 0
            empty_idx = None
            for idx in line:
                cell = board[idx]
                if cell == player:
                    count += 1
                elif cell == EMPTY:
                    empty_idx = idx
                else:
                    # 这条线上有对手棋子，不可能形成四连
                    count = -1
                    break
            if count == 3 and empty_idx is not None:
                winning_moves.append(empty_idx)
        return winning_moves

