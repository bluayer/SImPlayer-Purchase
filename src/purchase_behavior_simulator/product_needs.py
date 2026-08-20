from __future__ import annotations

from typing import Any, Iterable

from .models import Item, ProductNeedProfile


RATIONAL_TERMS = {
    "attack", "boost", "convenience", "currency", "defense", "functional",
    "hp", "magic", "progression", "speed", "subscription", "upgrade",
    "utility",
}
EMOTIONAL_TERMS = {
    "avatar", "collectible", "color", "cosmetic", "decoration", "emote",
    "emotional", "fashion", "identity", "skin", "social", "style", "vanity",
}


def _tokens(values: Iterable[Any]) -> set[str]:
    return {
        token.strip().lower()
        for value in values
        for token in str(value).replace("-", "_").split("_")
        if token.strip()
    }


def resolve_product_need_profile(item: Item) -> ProductNeedProfile:
    if item.need_profile is not None:
        return item.need_profile

    attributes = dict(item.attributes)
    rational_aspects = tuple(
        str(value)
        for value in attributes.get(
            "rational_aspects",
            attributes.get("functional_tags", ()),
        )
    )
    emotional_aspects = tuple(
        str(value)
        for value in attributes.get(
            "emotional_aspects",
            attributes.get("style_tags", ()),
        )
    )
    tokens = _tokens(
        (
            *item.categories,
            *rational_aspects,
            *emotional_aspects,
            attributes.get("utility_type", ""),
            attributes.get("style", ""),
        )
    )
    rational_hits = len(tokens.intersection(RATIONAL_TERMS))
    emotional_hits = len(tokens.intersection(EMOTIONAL_TERMS))
    if not rational_hits and not emotional_hits:
        return ProductNeedProfile(
            rational=0.5,
            emotional=0.5,
            source="inferred_unknown",
        )
    total = rational_hits + emotional_hits + 2.0
    return ProductNeedProfile(
        rational=(rational_hits + 1.0) / total,
        emotional=(emotional_hits + 1.0) / total,
        rational_aspects=rational_aspects,
        emotional_aspects=emotional_aspects,
        source="inferred_taxonomy",
    )
