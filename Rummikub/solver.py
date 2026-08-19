from typing import List, Tuple, Optional, Dict

# 颜色映射
COLOR_MAP = {'R': 0, 'Y': 1, 'B': 2, 'K': 3}
COLOR_NAMES = ['红', '黄', '蓝', '黑']
NUM_COLORS = 4
NUM_RANKS = 13  # 1-13 -> index 0-12
MAX_PER_CARD = 2  # 每种牌最多2张


def parse_hand(cards: List[str]) -> List[List[int]]:
    """将字符串牌组转换为 count[4][13] 计数矩阵"""
    count = [[0] * NUM_RANKS for _ in range(NUM_COLORS)]
    for card in cards:
        color = COLOR_MAP[card[0].upper()]
        rank = int(card[1:]) - 1  # 0-indexed
        if not (0 <= rank < NUM_RANKS):
            raise ValueError(f"非法牌号: {card}")
        count[color][rank] += 1
        if count[color][rank] > MAX_PER_CARD:
            raise ValueError(f"牌 {card} 超过最大数量2")
    return count


def find_min_card(count: List[List[int]]) -> Optional[Tuple[int, int]]:
    """找到当前手中最小的非零牌（先按数字，再按颜色）"""
    for r in range(NUM_RANKS):
        for c in range(NUM_COLORS):
            if count[c][r] > 0:
                return (c, r)
    return None


def get_valid_groups_for_anchor(count, anchor_color, anchor_rank):
    """
    生成所有包含锚点牌的合法组。
    返回: list of groups, 每个group是 [(color, rank), ...] 的列表
    """
    groups = []

    # === 类型1: 同色顺子，长度>=3，必须包含anchor ===
    c = anchor_color
    r = anchor_rank

    # 1. 从锚点向左扩展，找到连续段的左边界（包含）
    left = r
    while left > 0 and count[c][left - 1] > 0:
        left -= 1

    # 2. 从锚点向右扩展，找到连续段的右边界（不包含）
    right = r + 1
    while right < NUM_RANKS and count[c][right] > 0:
        right += 1

    # 3. [left, right) 即为包含锚点的最大同色连续段
    #    仅当该段长度 >= 3 时才有合法子顺子
    if right - left >= 3:
        # 4. 枚举所有包含锚点r、且长度>=3的子顺子[s, e)
        #    s 的取值范围: [left, r]
        #    e 的取值范围: [max(s+3, r+1), right]
        for start in range(left, r + 1):
            min_e = max(start + 3, r + 1)
            for e in range(min_e, right + 1):
                group = [(c, idx) for idx in range(start, e)]
                groups.append(group)

    # === 类型2: 同数字不同色，大小3或4，必须包含anchor ===
    r = anchor_rank
    available_colors = [c for c in range(NUM_COLORS) if count[c][r] > 0]
    if len(available_colors) >= 3:
        from itertools import combinations
        for size in (3, 4):
            for combo in combinations(available_colors, size):
                if anchor_color in combo:
                    group = [(c, r) for c in combo]
                    groups.append(group)

    return groups


def apply_group(count, group, delta):
    """对count矩阵施加+1或-1操作"""
    for c, r in group:
        count[c][r] += delta


def find_best_anchor(count: List[List[int]]) -> Optional[Tuple[int, int]]:
    """
    找到可选groups数量最少的锚点
    返回: (color, rank) 或 None（如果没有可用的牌）
    """
    # 收集所有可用的牌
    available_cards = []
    for r in range(NUM_RANKS):
        for c in range(NUM_COLORS):
            if count[c][r] > 0:
                available_cards.append((c, r))
    
    if not available_cards:
        return None
    
    # 对每个可用的牌计算其可选groups数量
    min_groups_count = float('inf')
    best_anchor = available_cards[0]  # 默认选择第一个
    
    for color, rank in available_cards:
        groups = get_valid_groups_for_anchor(count, color, rank)
        groups_count = len(groups)
        
        if groups_count < min_groups_count:
            min_groups_count = groups_count
            best_anchor = (color, rank)
            
            # 如果发现没有可选项，则立即返回（MRV启发式）
            if min_groups_count == 0:
                break
    
    return best_anchor


def solve(count: List[List[int]], path: List[List[Tuple[int, int]]]) -> bool:
    """回溯求解，path记录已选分组"""
    anchor = find_best_anchor(count)
    if anchor is None:
        return True  # 所有牌已分完

    groups = get_valid_groups_for_anchor(count, anchor[0], anchor[1])
    if not groups:
        # print(f"无法找到包含锚点牌 {COLOR_NAMES[anchor[0]]}{anchor[1]+1} 的合法组")
        return False  # 锚点牌无法归入任何组

    # 按组的长度排序，优先尝试较短的组合（通常更灵活）
    groups.sort(key=len)
    
    for group in groups:
        # print(f"尝试分组: {group}")
        apply_group(count, group, -1)
        path.append(group)

        if solve(count, path):
            return True

        else:
            path.pop()
            apply_group(count, group, +1)

    return False


def check_winnable(cards: List[str]) -> Optional[List[List[str]]]:
    """
    主入口：判断手牌是否可胜利。
    输入: ["R1", "R2", "R3", "Y5", "B5", "K5", ...]
    输出: 若可胜利，返回分组方案(字符串格式)；否则返回None
    """
    count = parse_hand(cards)

    # 快速剪枝：总牌数必须>=3
    total = sum(sum(row) for row in count)
    if total == 0:
        return []
    if total < 3:
        return None

    path = []
    if solve(count, path):
        # 转换回字符串格式
        result = []
        for group in path:
            result.append([f"{COLOR_NAMES[c]}{r+1}" for c, r in group])
        return result
    return None


# ==================== 测试示例 ====================
if __name__ == "__main__":
    # 测试1: 简单可胜利 (一个顺子 + 一个三张同数)
    hand1 = ["R1", "R2", "R3", "Y5", "B5", "K5"]
    res1 = check_winnable(hand1)
    print("测试1:", res1)

    # 测试2: 不可胜利
    hand2 = ["R1", "R2", "Y5", "B5", "K5"]
    res2 = check_winnable(hand2)
    print("测试2:", res2)

    # 测试3: 14张牌完整可胜利
    hand3 = (
        ["R1","R2","R3","R4",      # 红顺子4张
         "Y7","B7","K7",            # 7的同数3张
         "Y10","Y11","Y12","Y13",   # 黄顺子4张
         "R8","B8","K8"]            # 8的同数3张  → 共14张
    )
    res3 = check_winnable(hand3)
    print("测试3:", res3)