"""
四维四子棋 - 游戏状态管理模块
==========================

职责：
    1. 管理单个游戏房间的完整状态（棋盘、回合、胜负）
    2. 玩家注册与角色分配（支持人类/电脑混合）
    3. 落子校验与棋盘更新（拆分小函数，每函数 < 100 行）
    4. 提供可序列化的游戏快照（供 WebSocket 广播）

设计原则：
    - 不依赖 WebSocket 或任何网络层，保持纯逻辑
    - 所有公开方法返回值可 JSON 序列化
    - 通过 player.py 的 Player 类管理玩家身份与类型
"""

from typing import Dict, List, Optional

from .board import (
    EMPTY, PLAYER_1, PLAYER_2, PLAYER_3,
    TOTAL_CELLS, PLAYER_SYMBOLS,
    pos_2d_to_4d, pos_4d_to_index, pos_index_to_4d,
    check_win_at,
)
from .player import Player, HumanPlayer, ComputerPlayerRandom, ComputerPlayerNormal


class GameRoom:
    """
    单个四维四子棋游戏房间。

    属性：
        board:         长度为 256 的棋盘数组
        current_turn:  当前应落子的玩家编号
        move_count:    已落子总数
        game_over:     游戏是否已结束
        winner:        获胜者编号（0=无/平局）
        players:       活跃玩家 {player_id: Player}（start_game 时创建，reset 时销毁）
        _reservations: 座位预约 {slot: "human" | ("computer", "random"|"普通型")}
    """

    def __init__(self) -> None:
        self.board: List[int] = [EMPTY] * TOTAL_CELLS
        self.current_turn: int = PLAYER_1
        self.move_count: int = 0
        self.game_over: bool = False
        self.winner: int = EMPTY
        self.players: Dict[int, Player] = {}          # 游戏中活跃玩家
        self._reservations: Dict[int, object] = {}    # 座位预约
        self.started: bool = False
        self._last_move_index: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # 玩家管理（预约制：坐下/机器人 → 预约；start_game → 创建 Player）
    # ------------------------------------------------------------------

    def _slot_free(self, slot: int) -> bool:
        """指定座位是否空闲（未被预约）。"""
        return slot in (1, 2, 3) and slot not in self._reservations

    def sit_human(self, slot: int) -> bool:
        """人类预约座位。"""
        if not self._slot_free(slot):
            return False
        self._reservations[slot] = "human"
        return True

    def stand_up(self, slot: int) -> bool:
        """
        人类取消预约/离开游戏。

        - 游戏未开始：从 _reservations 中移除
        - 游戏进行中：从 players 中移除（Player 实例销毁）
        """
        # 游戏开始前：取消预约
        if self._reservations.get(slot) == "human":
            del self._reservations[slot]
            return True
        # 游戏进行中：移除活跃玩家
        p = self.players.get(slot)
        if p is not None and p.is_human:
            del self.players[slot]
            return True
        return False

    def add_robot(self, slot: int, strategy: str = "random") -> bool:
        """在座位放置机器人预约。"""
        if not self._slot_free(slot):
            return False
        self._reservations[slot] = ("computer", strategy)
        return True

    def remove_robot(self, slot: int) -> bool:
        """移除机器人预约。"""
        val = self._reservations.get(slot)
        if isinstance(val, tuple) and val[0] == "computer":
            del self._reservations[slot]
            return True
        return False

    def all_slots_filled(self) -> bool:
        """三个座位是否全部被预约。"""
        return len(self._reservations) >= 3

    def get_player_count(self) -> int:
        """已预约座位数（游戏开始前）或活跃玩家数（游戏中）。"""
        return len(self._reservations) if not self.started else len(self.players)

    def is_computer_turn(self) -> bool:
        """当前回合是否为电脑玩家。"""
        p = self.players.get(self.current_turn)
        return p is not None and not p.is_human

    def get_player(self, player_id: int) -> Optional[Player]:
        """获取活跃玩家对象。"""
        return self.players.get(player_id)

    def start_game(self) -> dict:
        """
        开始游戏：根据预约创建 Player 实例并开始对局。
        """
        if not self.all_slots_filled():
            return {"success": False, "message": "需要三位玩家才能开始游戏"}
        if self.started:
            return {"success": False, "message": "游戏已经开始"}

        # 根据预约创建 Player 实例
        self.players.clear()
        for slot, reservation in self._reservations.items():
            if reservation == "human":
                self.players[slot] = HumanPlayer(slot)
            elif isinstance(reservation, tuple) and reservation[0] == "computer":
                _, strategy = reservation
                if strategy == "普通型":
                    self.players[slot] = ComputerPlayerNormal(slot)
                else:
                    self.players[slot] = ComputerPlayerRandom(slot)

        self.started = True
        return {"success": True, "message": "游戏开始！"}

    # ------------------------------------------------------------------
    # 落子逻辑（拆分为 3 个小函数，每函数 < 100 行）
    # ------------------------------------------------------------------

    def make_move(self, player_id: int, row: int, col: int) -> dict:
        """
        处理一次落子请求（对外统一入口）。

        返回 dict: {success, message, game_over, winner, move?}
        """
        error = self._validate_move(player_id, row, col)
        if error:
            return error

        w, x, y, z, index = self._place_piece(player_id, row, col)
        return self._finish_turn(player_id, row, col, w, x, y, z, index)

    def _validate_move(
        self, player_id: int, row: int, col: int
    ) -> Optional[dict]:
        """
        落子前置校验。通过返回 None，失败返回错误 dict。
        """
        if self.game_over:
            return self._error("游戏已结束，请等待新一局", self.winner)

        if not self.started:
            return self._error("游戏尚未开始")

        if player_id != self.current_turn:
            return self._error(f"当前不是玩家 {player_id} 的回合")

        if not (0 <= row < 16 and 0 <= col < 16):
            return self._error("坐标超出棋盘范围")

        w, x, y, z = pos_2d_to_4d(row, col)
        index = pos_4d_to_index(w, x, y, z)
        if self.board[index] != EMPTY:
            return self._error("该位置已有棋子")

        return None  # 校验通过

    def _place_piece(
        self, player_id: int, row: int, col: int
    ) -> tuple:
        """在棋盘上放置棋子，返回 (w, x, y, z, index)。同时记录到玩家。"""
        w, x, y, z = pos_2d_to_4d(row, col)
        index = pos_4d_to_index(w, x, y, z)
        self.board[index] = player_id
        self.move_count += 1
        self._last_move_index[player_id] = index
        # 记录到玩家实例
        player = self.players.get(player_id)
        if player is not None:
            player.record_move(row, col)
        return (w, x, y, z, index)

    def _finish_turn(
        self, player_id: int, row: int, col: int,
        w: int, x: int, y: int, z: int, index: int,
    ) -> dict:
        """落子后的胜负/平局检测与回合切换。"""
        # 胜负检测
        if check_win_at(self.board, index, player_id):
            self.game_over = True
            self.winner = player_id
            return self._build_result(
                success=True,
                message=f"玩家 {player_id} ({PLAYER_SYMBOLS[player_id]}) 获胜！",
                game_over=True, winner=player_id,
                player=player_id, row=row, col=col,
                w=w, x=x, y=y, z=z, index=index,
            )

        # 平局检测
        if self.move_count >= TOTAL_CELLS:
            self.game_over = True
            self.winner = EMPTY
            return self._build_result(
                success=True,
                message="棋盘已满，平局！",
                game_over=True, winner=EMPTY,
                player=player_id, row=row, col=col,
                w=w, x=x, y=y, z=z, index=index,
            )

        # 正常切换回合：1→2→3→1…
        self.current_turn = (self.current_turn % 3) + 1
        return self._build_result(
            success=True,
            message=f"玩家 {player_id} 落子成功，轮到玩家 {self.current_turn}",
            game_over=False, winner=EMPTY,
            player=player_id, row=row, col=col,
            w=w, x=x, y=y, z=z, index=index,
        )


    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _error(message: str, winner: int = EMPTY) -> dict:
        """构建校验失败的错误返回。"""
        return {
            "success": False,
            "message": message,
            "game_over": winner != EMPTY,
            "winner": winner,
        }

    @staticmethod
    def _build_result(
        success: bool, message: str, game_over: bool, winner: int,
        player: int, row: int, col: int,
        w: int, x: int, y: int, z: int, index: int,
    ) -> dict:
        """构建统一的落子操作返回字典。"""
        return {
            "success": success,
            "message": message,
            "game_over": game_over,
            "winner": winner,
            "move": {
                "player": player,
                "row": row, "col": col,
                "w": w, "x": x, "y": y, "z": z,
                "index": index,
            },
        }

    # ------------------------------------------------------------------
    # 状态快照
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """
        返回当前游戏状态的完整快照。

        - 游戏开始前：player 信息来自 _reservations
        - 游戏开始后：player 信息来自 self.players
        """
        last_moves_2d: Dict[int, list] = {}
        for pid, idx in self._last_move_index.items():
            _w, _x, _y, _z = pos_index_to_4d(idx)
            last_moves_2d[pid] = [4 * _w + _x, 4 * _y + _z]

        player_types: Dict[int, str] = {}
        player_strategies: Dict[int, str] = {}
        player_count: int = 0

        if self.started:
            for pid, pl in self.players.items():
                player_types[pid] = "human" if pl.is_human else "computer"
                player_strategies[pid] = _class_to_strategy(pl)
            player_count = len(self.players)
        else:
            for slot, reservation in self._reservations.items():
                if reservation == "human":
                    player_types[slot] = "human"
                    player_strategies[slot] = ""
                elif isinstance(reservation, tuple) and reservation[0] == "computer":
                    player_types[slot] = "computer"
                    player_strategies[slot] = reservation[1]  # "random" or "普通型"
            player_count = len(self._reservations)

        return {
            "board": self.board,
            "current_turn": self.current_turn,
            "move_count": self.move_count,
            "game_over": self.game_over,
            "winner": self.winner,
            "started": self.started,
            "player_count": player_count,
            "player_types": player_types,
            "player_strategies": player_strategies,
            "last_moves": last_moves_2d,
        }

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """
        重置游戏（销毁 Player 实例，保留座位预约）。

        Player 实例在游戏结束时被销毁，符合「仅在游戏期间存在」的设计。
        """
        self.board = [EMPTY] * TOTAL_CELLS
        self.current_turn = PLAYER_1
        self.move_count = 0
        self.game_over = False
        self.winner = EMPTY
        self.started = False
        self.players.clear()          # 销毁 Player 实例
        self._last_move_index = {}
        return self.get_state()


def _class_to_strategy(player: Player) -> str:
    """从 Player 子类推导策略名。"""
    from .player import ComputerPlayerRandom, ComputerPlayerNormal
    if isinstance(player, ComputerPlayerRandom):
        return "随机型"
    if isinstance(player, ComputerPlayerNormal):
        return "普通型"
    return ""
