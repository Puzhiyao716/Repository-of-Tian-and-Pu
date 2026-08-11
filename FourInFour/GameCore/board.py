"""
四维四子棋 - 棋盘核心模块
=====================

职责：
    1. 定义棋盘常量（尺寸、玩家符号）
    2. 4D 坐标 (w, x, y, z) 与 2D 屏幕坐标 (row, col) 之间的双向映射
    3. 预计算所有可能的获胜线（4D 空间中所有合法的四子直线）
    4. 落子后仅增量检测新落子处是否产生胜负

设计原则：
    - 本模块不持有棋盘状态（状态由 game.py 管理），仅提供纯函数式的工具方法
    - 所有获胜线在模块加载时一次性预计算并缓存，之后只做 O(1) 查表
"""

from typing import List, Tuple, Set

# ============================================================================
# 常量定义
# ============================================================================

# 每个维度的尺寸（4×4×4×4）
BOARD_SIZE: int = 4

# 棋盘总位置数
TOTAL_CELLS: int = BOARD_SIZE ** 4  # 256

# 玩家编号（0 表示空位）
EMPTY: int = 0
PLAYER_1: int = 1
PLAYER_2: int = 2
PLAYER_3: int = 3

# 三位玩家的 UI 显示符号
PLAYER_SYMBOLS: dict = {
    PLAYER_1: "○",   # 圆圈
    PLAYER_2: "✕",   # 叉号
    PLAYER_3: "△",   # 三角
}


# ============================================================================
# 坐标映射
# ============================================================================

def pos_4d_to_2d(w: int, x: int, y: int, z: int) -> Tuple[int, int]:
    """
    将 4D 坐标 (w, x, y, z) 映射为 16×16 屏幕上的 (row, col)。

    映射公式：
        row = 4 * w + x      (范围 0~15)
        col = 4 * y + z      (范围 0~15)
    """
    return (4 * w + x, 4 * y + z)


def pos_2d_to_4d(row: int, col: int) -> Tuple[int, int, int, int]:
    """
    将 16×16 屏幕上的 (row, col) 反向映射为 4D 坐标 (w, x, y, z)。

    逆映射公式：
        w = row // 4       (该位置所属的小棋盘行索引)
        x = row % 4        (小棋盘内的行偏移)
        y = col // 4       (小棋盘列索引)
        z = col % 4        (小棋盘内的列偏移)
    """
    return (row // 4, row % 4, col // 4, col % 4)


# ============================================================================
# 四维坐标的一维索引（用于快速数组存取）
# ============================================================================

def pos_4d_to_index(w: int, x: int, y: int, z: int) -> int:
    """
    将 (w, x, y, z) 压平为一维数组索引。
    采用行优先：index = w * 64 + x * 16 + y * 4 + z （范围 0~255）
    """
    return w * 64 + x * 16 + y * 4 + z


def pos_index_to_4d(index: int) -> Tuple[int, int, int, int]:
    """将一维索引反向展开为 (w, x, y, z)。"""
    w = index // 64
    remainder = index % 64
    x = remainder // 16
    remainder = remainder % 16
    y = remainder // 4
    z = remainder % 4
    return (w, x, y, z)


# ============================================================================
# 获胜线预计算
# ============================================================================

def _generate_all_winning_lines() -> List[List[int]]:
    """
    预计算 4D 棋盘上所有可能的获胜线。

    获胜线定义：
        4 个位置沿某个方向向量 (dw, dx, dy, dz) 等距排列，
        且均在棋盘范围内。方向向量各分量 ∈ {-1, 0, 1} 且不全为 0。

    返回：
        二维列表，每个元素是一条获胜线，存储该线上 4 个位置的一维索引。
    """
    # 步骤1：生成所有 80 个合法方向向量 (3^4 - 1 = 80)
    directions: List[Tuple[int, int, int, int]] = []
    for dw in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dw == 0 and dx == 0 and dy == 0 and dz == 0:
                        continue
                    directions.append((dw, dx, dy, dz))

    # 步骤2：枚举所有 (起点 × 方向)，收集合法线段
    lines: List[List[int]] = []
    seen: Set[frozenset] = set()  # 用 frozenset 去重

    for w in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                for z in range(BOARD_SIZE):
                    for dw, dx, dy, dz in directions:
                        indices = []
                        valid = True
                        for step in range(4):
                            nw = w + step * dw
                            nx = x + step * dx
                            ny = y + step * dy
                            nz = z + step * dz
                            if not (0 <= nw < BOARD_SIZE and
                                    0 <= nx < BOARD_SIZE and
                                    0 <= ny < BOARD_SIZE and
                                    0 <= nz < BOARD_SIZE):
                                valid = False
                                break
                            indices.append(pos_4d_to_index(nw, nx, ny, nz))

                        if not valid:
                            continue

                        key = frozenset(indices)
                        if key not in seen:
                            seen.add(key)
                            lines.append(indices)

    return lines


# 模块加载时一次性预计算，后续只读取
WINNING_LINES: List[List[int]] = _generate_all_winning_lines()

# ============================================================================
# 索引 → 获胜线映射（用于加速增量式检测）
# ============================================================================

def _build_lines_by_cell() -> List[List[List[int]]]:
    """
    构建"位置 → 经过该位置的获胜线列表"的映射。
    供增量式胜负检测使用：落子后只检查经过该位置的线。
    """
    lines_by_cell: List[List[List[int]]] = [[] for _ in range(TOTAL_CELLS)]
    for line in WINNING_LINES:
        for idx in line:
            lines_by_cell[idx].append(line)
    return lines_by_cell


LINES_BY_CELL: List[List[List[int]]] = _build_lines_by_cell()
# print("[TESTINFO] 预计算完成：总线数：", len(WINNING_LINES), "总位置数：", len(LINES_BY_CELL))
# print(f"[TESTINFO] WINNING_LINES : {type(WINNING_LINES)}, LINES_BY_CELL : {type(LINES_BY_CELL)}")
# print(f"[TESTINFO] WINNING_LINES : {type(WINNING_LINES[0])}, LINES_BY_CELL : {type(LINES_BY_CELL[0])}")
# print(f"[TESTINFO] WINNING_LINES : {len(WINNING_LINES[0])}, LINES_BY_CELL : {len(LINES_BY_CELL[0])}")

# ============================================================================
# 胜负检测（增量式）
# ============================================================================

def check_win_at(
    board: List[int],
    last_move_index: int,
    player: int,
    potential_win: bool = False
):
    """
    增量式胜负检测：仅检测最后落子处是否达成四连；可选同时检测必胜点。

    参数：
        board:           长度为 256 的一维棋盘数组（0=空，1/2/3=玩家）
        last_move_index: 最后落子的一维索引
        player:          当前落子的玩家编号
        potential_win:   False → 仅检测胜负，返回 bool
                         True  → 同时检测至多两个必胜点，返回 (bool, list)

    返回（potential_win=False）：
        bool — True 表示该玩家已四连获胜

    返回（potential_win=True）：
        (has_won: bool, pot_points: List[List[int]])
        has_won    — True 表示该玩家已四连获胜（此时 pot_points 为空）
        pot_points — 必胜点 2D 坐标列表 [[row, col], ...]，至多两个
    """
    lines = LINES_BY_CELL[last_move_index]

    # ---- 快速路径：仅检测胜负（默认行为，向后兼容）----
    if not potential_win:
        for line in lines:
            i0, i1, i2, i3 = line
            c0, c1, c2, c3 = board[i0], board[i1], board[i2], board[i3]
            if c0 == player and c1 == player and c2 == player and c3 == player:
                return True
        return False

    # ---- 完整路径：检测胜负 + 至多两个必胜点 ----
    pot_points: List[List[int]] = []

    for line in lines:
        i0, i1, i2, i3 = line
        c0, c1, c2, c3 = board[i0], board[i1], board[i2], board[i3]

        # 三子已占 → 检查剩余一格是否为空（即必胜点）
        cnt = (c0 == player) + (c1 == player) + (c2 == player) + (c3 == player)
        if cnt == 4:
            # 四连 → 直接获胜，无需关心必胜点
            return True, []
        
        elif cnt == 3:
            # 精确定位空位（必须显式验证 cX == EMPTY，排除对手占据的情况）
            empty_idx: int = -1
            if c0 == EMPTY:      empty_idx = i0
            elif c1 == EMPTY:    empty_idx = i1
            elif c2 == EMPTY:    empty_idx = i2
            elif c3 == EMPTY:    empty_idx = i3
            if empty_idx == -1:
                continue          # 被对手阻挡，非必胜点

            # 转 2D 坐标（内联避免函数调用）
            _w = empty_idx // 64
            _r = empty_idx % 64
            _x = _r // 16
            _r = _r % 16
            _y = _r // 4
            _z = _r % 4
            pot_points.append([4 * _w + _x, 4 * _y + _z])

            if len(pot_points) >= 4:
                break

    return False, pot_points


def get_winning_line(
    board: List[int],
    last_move_index: int,
    player: int
) -> List[int]:
    """
    获取获胜线的 4 个格子一维索引。

    参数同 check_win_at。

    返回：
        获胜线的 4 个一维索引列表；若未获胜则返回空列表。
    """
    for line in LINES_BY_CELL[last_move_index]:
        if all(board[idx] == player for idx in line):
            return list(line)
    return []
