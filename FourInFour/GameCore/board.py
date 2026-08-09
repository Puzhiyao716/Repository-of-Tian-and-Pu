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


# ============================================================================
# 胜负检测（增量式）
# ============================================================================

def check_win_at(
    board: List[int],
    last_move_index: int,
    player: int
) -> bool:
    """
    增量式胜负检测：仅检测最后落子处是否达成四连。

    参数：
        board:           长度为 256 的一维棋盘数组（0=空，1/2/3=玩家）
        last_move_index: 最后落子的一维索引
        player:          当前落子的玩家编号

    返回：
        True 表示该玩家获胜
    """
    for line in LINES_BY_CELL[last_move_index]:
        if all(board[idx] == player for idx in line):
            return True
    return False
