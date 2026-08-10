"""
四维四子棋 - 玩家模块
===================

类层次：
    Player (ABC)       — 抽象基类，含落子记录
    ├── HumanPlayer    — 人类玩家（WebSocket 输入）
    ├── ComputerPlayerRandom   — 电脑·随机型（随机落子）
    └── ComputerPlayerNormal   — 电脑·普通型（MCTS + UCB1）

设计原则：
    - 每种电脑类型是独立类，内部实现各自的 choose_move 逻辑
    - Player 实例只在 start_game 时创建，reset 时销毁
    - 每个玩家记录自己所有落子位置
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import random

from .board import PLAYER_SYMBOLS, pos_2d_to_4d, pos_4d_to_index


# ============================================================================
# Player 抽象基类
# ============================================================================

class Player(ABC):
    """
    玩家抽象基类。

    属性：
        player_id : int              — 玩家编号（1/2/3）
        symbol    : str              — 显示符号（○/✕/△）
        moves     : List[Tuple[int, int]] — 本局所有落子 (row, col)
    """

    def __init__(self, player_id: int) -> None:
        if player_id not in (1, 2, 3):
            raise ValueError(f"玩家编号必须为 1/2/3，收到：{player_id}")
        self.player_id: int = player_id
        self.symbol: str = PLAYER_SYMBOLS[player_id]
        self.moves: List[Tuple[int, int]] = []   # 本局落子记录

    @property
    @abstractmethod
    def is_human(self) -> bool:
        """是否为人类玩家。"""
        ...

    def record_move(self, row: int, col: int) -> None:
        """记录一步落子。"""
        self.moves.append((row, col))

    def clear_moves(self) -> None:
        """清空落子记录。"""
        self.moves.clear()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.player_id}, symbol={self.symbol})"


# ============================================================================
# HumanPlayer
# ============================================================================

class HumanPlayer(Player):
    """人类玩家 —— 落子由 WebSocket 消息驱动。"""

    @property
    def is_human(self) -> bool:
        return True


# ============================================================================
# ComputerPlayerRandom（随机型）
# ============================================================================

class ComputerPlayerRandom(Player):
    """随机型电脑 —— 在所有合法空位中均匀随机选择。"""

    @property
    def is_human(self) -> bool:
        return False

    def choose_move(self, board: List[int]) -> Tuple[int, int]:
        """随机选择一个合法空位。"""
        empty = [i for i, v in enumerate(board) if v == 0]
        if not empty:
            raise RuntimeError("棋盘已满，无法落子")
        idx = random.choice(empty)
        return _index_to_2d(idx)


# ============================================================================
# ComputerPlayerNormal（普通型 / MCTS）
# ============================================================================

class ComputerPlayerNormal(Player):
    """
    普通型电脑 —— 蒙特卡洛树搜索 + UCB1。

    以自己赢为第一目的，假设所有对手也各自为赢而奋斗。
    MCTS 引擎惰性初始化。
    """

    def __init__(self, player_id: int,
                 time_limit: float = 5.0, max_iters: int = 10000) -> None:
        super().__init__(player_id)
        self._mcts_engine = None  # 惰性初始化
        self._time_limit = time_limit
        self._max_iters = max_iters
        self._last_thinking_time: float = 0.0
        self._last_thinking_iters: int = 0

    @property
    def is_human(self) -> bool:
        return False

    @property
    def thinking_stats(self) -> dict:
        """返回最近一次思考的统计信息。"""
        return {
            "time": round(self._last_thinking_time, 3),
            "iters": self._last_thinking_iters,
        }

    def choose_move(self, board: List[int]) -> Tuple[int, int]:
        """使用 MCTS 搜索选择落子。"""
        if self._mcts_engine is None:
            from FourInFour.AI.mcts import MCTSEngine
            self._mcts_engine = MCTSEngine(
                player_id=self.player_id,
                time_limit=self._time_limit,
                max_iters=self._max_iters,
            )
        (row, col), elapsed, iters = self._mcts_engine.choose_move(
            board, self.player_id)
        self._last_thinking_time = elapsed
        self._last_thinking_iters = iters
        return (row, col)


# ============================================================================
# 工具
# ============================================================================

def _index_to_2d(idx: int) -> Tuple[int, int]:
    """一维索引 → (row, col)。"""
    _w = idx // 64
    r = idx % 64
    _x = r // 16
    r = r % 16
    _y = r // 4
    _z = r % 4
    return (4 * _w + _x, 4 * _y + _z)

