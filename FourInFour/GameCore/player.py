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
from typing import List, Tuple, Dict
import random

from .board import PLAYER_SYMBOLS, pos_2d_to_4d, pos_4d_to_index


# ============================================================================
# Player 抽象基类
# ============================================================================

class _Player(ABC):
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

class HumanPlayer(_Player):
    """人类玩家 —— 落子由 WebSocket 消息驱动。"""

    @property
    def is_human(self) -> bool:
        return True


# ============================================================================
# ComputerPlayerRandom（随机型）
# ============================================================================

class ComputerPlayerRandom(_Player):
    """随机型电脑 —— 在所有合法空位中均匀随机选择。"""

    @property
    def is_human(self) -> bool:
        return False

    def choose_move(self, board: List[int],
                    last_moves: dict = None) -> Tuple[int, int]:
        """随机选择一个合法空位。last_moves 参数为接口兼容，本策略不使用。"""
        empty = [i for i, v in enumerate(board) if v == 0]
        if not empty:
            raise RuntimeError("棋盘已满，无法落子")
        idx = random.choice(empty)
        return _index_to_2d(idx)


# ============================================================================
# ComputerPlayerNormal（普通型 / MCTS）
# ============================================================================

class _ComputerPlayer(_Player):
    """电脑玩家基类 —— MCTS 引擎公共逻辑（惰性初始化、统计、落子追踪）。"""

    def __init__(self, player_id: int,
                 time_limit: float = 5.0, max_iters: int = 10000) -> None:
        super().__init__(player_id)
        self._engine = None  # 惰性初始化
        self._time_limit = time_limit
        self._max_iters = max_iters
        self._last_thinking_time: float = 0.0
        self._last_thinking_iters: int = 0
        # 追踪所有玩家的最近一步棋：{player_id: 一维索引}
        self._last_move_index: Dict[int, int] = {}

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

    def record_move(self, row: int, col: int) -> None:
        """记录落子（同时转为一维索引供 MCTS 使用）。"""
        super().record_move(row, col)
        w, x, y, z = pos_2d_to_4d(row, col)
        idx = pos_4d_to_index(w, x, y, z)
        self._last_move_index[self.player_id] = idx

    def clear_moves(self) -> None:
        """清空落子记录（同时清空索引追踪）。"""
        super().clear_moves()
        self._last_move_index.clear()

    def _get_engine(self):
        """子类重写以提供具体的 MCTS 引擎。"""
        raise NotImplementedError

    def choose_move(self, board: List[int],
                    last_moves: dict = None) -> Tuple[int, int, int, int]:
        """
        使用 MCTS 搜索选择落子。

        last_moves: {player_id: 一维索引}，所有玩家最近一步棋。
        返回: (w, x, y, z) 四维坐标
        """
        if self._engine is None:
            self._engine = self._get_engine()
        # 合并内部追踪的 last_moves 与外部传入的 last_moves
        merged = {}
        if last_moves:
            merged.update(last_moves)
        merged.update(self._last_move_index)
        move_data, elapsed, iters = self._engine.choose_move(
            board, self.player_id, merged)
        self._last_thinking_time = elapsed
        self._last_thinking_iters = iters
        return move_data


class ComputerPlayerNormal(_ComputerPlayer):
    """
    普通型电脑 —— 蒙特卡洛树搜索 + UCB1。

    以自己赢为第一目的，假设所有对手也各自为赢而奋斗。
    """

    def _get_engine(self):
        from FourInFour.AI.mcts import MCTSEngine
        return MCTSEngine(
            player_id=self.player_id,
            time_limit=self._time_limit,
            max_iters=self._max_iters,
        )


class NormalTian(_ComputerPlayer):
    """
    Tian型电脑 —— MCTS + UCB1 + Alphabet 剪枝。

    在标准 MCTS 的基础上，通过必胜点分析实现搜索剪枝：
    - 自己有必胜点 → 只探索必胜分支
    - 对手有必胜点 → 优先堵截
    """

    def _get_engine(self):
        from FourInFour.AI import ABMCTSEngine
        return ABMCTSEngine(
            player_id=self.player_id,
            time_limit=self._time_limit,
            max_iters=self._max_iters,
        )


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

