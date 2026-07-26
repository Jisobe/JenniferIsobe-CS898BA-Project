from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

CATEGORIES = [
    "ones", "twos", "threes", "fours", "fives", "sixes",
    "three_of_a_kind", "four_of_a_kind", "full_house",
    "small_straight", "large_straight", "yahtzee", "chance",
]
UPPER_CATEGORIES = {"ones", "twos", "threes", "fours", "fives", "sixes"}
UPPER_CATEGORIES_TO_FACE = {
    "ones": 1, "twos": 2, "threes": 3, "fours": 4, "fives": 5, "sixes": 6,
}
UPPER_BONUS_THRESHOLD = 63
UPPER_BONUS_VALUE = 35
YAHTZEE_VALUE = 50
YAHTZEE_BONUS_VALUE = 100

def _validate_dice(dice, expected_dice_count=5):
    if len(dice) != expected_dice_count:
        raise ValueError(f"Expected {expected_dice_count} dice values, got {len(dice)}")
    for die in dice:
        if die not in (1, 2, 3, 4, 5, 6):
            raise ValueError(f"Invalid die value: {die} (must be 1-6)")

def score_category(dice, category, count=5):
    _validate_dice(dice, count)
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")

    counts = Counter(dice)
    total = sum(dice)

    if category in UPPER_CATEGORIES:
        face = UPPER_CATEGORIES_TO_FACE[category]
        return face * counts.get(face, 0)

    if category == "three_of_a_kind":
        return total if max(counts.values()) >= 3 else 0

    if category == "four_of_a_kind":
        return total if max(counts.values()) >= 4 else 0

    if category == "full_house":
        values = sorted(counts.values())
        return 25 if values == [2, 3] else 0

    if category == "small_straight":
        distinct = set(dice)
        runs = [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]
        return 30 if any(run.issubset(distinct) for run in runs) else 0

    if category == "large_straight":
        distinct = set(dice)
        return 40 if distinct in ({1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}) else 0

    if category == "yahtzee":
        return YAHTZEE_VALUE if max(counts.values()) == 5 else 0

    if category == "chance":
        return total

    raise AssertionError("unreachable")

def all_category_scores(dice):
    return {cat: score_category(dice, cat) for cat in CATEGORIES}

@dataclass
class Scorecard:
    entries = field(
        default_factory=lambda: {cat: None for cat in CATEGORIES}
    )
    bonus_yahtzee_count: int = 0

    def is_open(self, category):
        return self.entries.get(category) is None

    def open_categories(self):
        return [c for c in CATEGORIES if self.is_open(c)]

    def record(self, category, dice):
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        if not self.is_open(category):
            raise ValueError(f"Category '{category}' is already filled")

        points = score_category(dice, category)
        if category != "yahtzee" and max(Counter(dice).values()) == 5 and self.entries.get("yahtzee") == YAHTZEE_VALUE:
            self.bonus_yahtzee_count += 1

        self.entries[category] = points
        return points

    def scratch(self, category):
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        if not self.is_open(category):
            raise ValueError(f"Category '{category}' is already filled")
        self.entries[category] = 0

    def upper_subtotal(self):
        return sum(self.entries[c] or 0 for c in UPPER_CATEGORIES)

    def upper_bonus(self):
        return UPPER_BONUS_VALUE if self.upper_subtotal() >= UPPER_BONUS_THRESHOLD else 0

    def lower_subtotal(self):
        lower = [c for c in CATEGORIES if c not in UPPER_CATEGORIES]
        return sum(self.entries[c] or 0 for c in lower)

    def yahtzee_bonus_total(self):
        return self.bonus_yahtzee_count * YAHTZEE_BONUS_VALUE

    def grand_total(self):
        return (
            self.upper_subtotal()
            + self.upper_bonus()
            + self.lower_subtotal()
            + self.yahtzee_bonus_total()
        )

    def is_complete(self):
        return all(v is not None for v in self.entries.values())

if __name__ == "__main__":
    sc = Scorecard()
    print(all_category_scores([3, 3, 3, 5, 5]))
    print(sc.record("full_house", [3, 3, 3, 5, 5]))
    print(sc.record("threes", [3, 3, 6, 1, 2]))
    print(sc.grand_total())