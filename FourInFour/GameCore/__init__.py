"""
四维四子棋 - GameCore 核心逻辑包

统一导出所有对外接口，外部模块只需：
    from FourInFour.GameCore import GameRoom, PLAYER_SYMBOLS, ...
"""

from .board import (
    BOARD_SIZE, TOTAL_CELLS,
    EMPTY, PLAYER_1, PLAYER_2, PLAYER_3,
    PLAYER_SYMBOLS,
    pos_4d_to_2d, pos_2d_to_4d,
    pos_4d_to_index, pos_index_to_4d,
    check_win_at, get_winning_line, WINNING_LINES, LINES_BY_CELL,
)
from .player import _Player, HumanPlayer, ComputerPlayerRandom, ComputerPlayerNormal, NormalTian
from .game import GameRoom

__all__ = [
    # 常量
    "BOARD_SIZE", "TOTAL_CELLS",
    "EMPTY", "PLAYER_1", "PLAYER_2", "PLAYER_3",
    "PLAYER_SYMBOLS",
    # 坐标映射
    "pos_4d_to_2d", "pos_2d_to_4d",
    "pos_4d_to_index", "pos_index_to_4d",
    # 胜负检测
    "check_win_at", "get_winning_line", "WINNING_LINES", "LINES_BY_CELL",
    # 玩家
    "_Player", "HumanPlayer", "ComputerPlayerRandom", "ComputerPlayerNormal", "NormalTian",
    # 游戏房间
    "GameRoom",
]