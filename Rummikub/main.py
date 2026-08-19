import random
import sys
import time
from solver import check_winnable

# ==================== 常量与工具函数 ====================
COLORS = ['R', 'Y', 'B', 'K']
COLOR_DISPLAY = {'R': '红', 'Y': '黄', 'B': '蓝', 'K': '黑'}
RANKS = list(range(1, 14))  # 1-13


def build_deck() -> list[str]:
    """构建104张牌的完整牌堆"""
    deck = []
    for color in COLORS:
        for rank in RANKS:
            # 每种颜色每个数字各2张
            deck.append(f"{color}{rank}")
            deck.append(f"{color}{rank}")
    return deck


def format_card(card: str) -> str:
    """将内部编码转换为人类可读格式，如 R1 -> 红1"""
    color = COLOR_DISPLAY.get(card[0], card[0])
    rank = card[1:]
    return f"{color}{rank}"


def display_hand(hand: list[str]) -> None:
    """美观地展示当前手牌"""
    sorted_hand = sorted(hand, key=lambda c: (int(c[1:]), COLORS.index(c[0])))
    print("\n" + "=" * 50)
    print(f"[扑克] 当前手牌 ({len(sorted_hand)} 张):")
    display_cards = [format_card(c) for c in sorted_hand]
    # 每行显示7张，便于阅读
    for i in range(0, len(display_cards), 7):
        row = display_cards[i:i + 7]
        print("  " + "  ".join(row))
    print("=" * 50)


def display_solution(solution: list[list[str]]) -> None:
    """展示胜利的分组方案"""
    print("\n[恭喜] 当前手牌可以完美分组！")
    print(f"[统计] 总牌数: {sum(len(g) for g in solution)} 张 | 共 {len(solution)} 组\n")
    for i, group in enumerate(solution, 1):
        formatted = [format_card(c) for c in group]
        print(f"  第 {i} 组 ({len(group)}张): {', '.join(formatted)}")
    print()


# ==================== 主游戏循环 ====================
def main():
    print("╔══════════════════════════════════════╗")
    print("║      卡牌完美分组 - 单人挑战         ║")
    print("╚══════════════════════════════════════╝")
    print("规则提示: 顺子需同色连续(≥3张)，同数需异色(3或4张)")
    print("输入 'q' 可随时退出游戏\n")

    # 初始化牌堆并发牌
    deck = build_deck()
    random.shuffle(deck)
    hand = [deck.pop() for _ in range(14)]

    while True:
        display_hand(hand)

        # 判断是否可胜利
        start_time = time.time()
        solution = check_winnable(hand)
        elapsed_time = time.time() - start_time
        print(f"[计时] checkwin函数耗时: {elapsed_time:.3f} 秒")

        if solution is not None:
            display_solution(solution)
            print("[成功] 你已达成胜利条件！游戏结束！")
            break
        else:
            print("\n[失败] 当前手牌无法完美分组。")
            user_input = input("[回车] 按下回车来摸牌 (输入 q 退出): ").strip()

            if user_input.lower() == 'q':
                print("\n[再见] 感谢游玩，再见！")
                sys.exit(0)

            # 摸牌逻辑
            if len(deck) == 0:
                print("\n[警告] 牌堆已空，无法继续摸牌！游戏结束。")
                break

            new_card = deck.pop()
            hand.append(new_card)
            print(f"\n[摸牌] 摸到一张牌: {format_card(new_card)}")


if __name__ == "__main__":
    main()