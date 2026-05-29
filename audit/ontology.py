from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

SECRET_PATTERNS: dict[str, list[str]] = {
    "child_present": [r"\bchild\b", r"\bbaby\b", r"\btoddler\b", r"\bkid\b", r"\bgirl\b", r"\bboy\b", r"\bchildren\b"],
    "medical_context": [r"\bhospital\b", r"\bclinic\b", r"\bmedical\b", r"\bdoctor\b", r"\bnurse\b", r"\bpatient\b", r"\bstethoscope\b", r"\bwheelchair\b"],
    "home_context": [r"\bkitchen\b", r"\bbedroom\b", r"\bliving room\b", r"\bcouch\b", r"\bsofa\b", r"\bhome\b", r"\bhouse\b", r"\bbathroom\b"],
    "work_or_office_context": [r"\boffice\b", r"\bdesk\b", r"\blaptop\b", r"\bmonitor\b", r"\bkeyboard\b", r"\bmeeting\b", r"\bcubicle\b", r"\bworkstation\b"],
    "food_or_meal_context": [r"\bfood\b", r"\bmeal\b", r"\bplate\b", r"\brestaurant\b", r"\bdining\b", r"\bbreakfast\b", r"\blunch\b", r"\bdinner\b", r"\bcake\b", r"\bpizza\b"],
}


def assign_secret_labels(text: str) -> dict[str, int]:
    text = (text or "").lower()
    labels = {}
    for secret, patterns in SECRET_PATTERNS.items():
        labels[secret] = int(any(re.search(p, text) for p in patterns))
    return labels


def extract_coarse_tags(texts: Iterable[str]) -> Counter:
    c = Counter()
    for text in texts:
        labels = assign_secret_labels(text)
        for k, v in labels.items():
            if v:
                c[k] += 1
    return c
