from __future__ import annotations

import re
from dataclasses import replace

from models import DocumentImage, Slide, SlideDeck


def attach_images(deck: SlideDeck, images: tuple[DocumentImage, ...]) -> SlideDeck:
    """Adds images to a slide

    Args:
        deck (SlideDeck): slide deck
        images (tuple[DocumentImage, ...]): images to add

    Returns:
        SlideDeck: updated slide deck
    """
    if not images:
        return deck

    unused = list(images)
    updated: list[Slide] = []
    for slide in deck.slides:
        image = _best_image(slide, unused)
        if image is None:
            updated.append(slide)
            continue
        unused.remove(image)
        updated.append(replace(slide, image_path=image.path))
    return replace(deck, slides=tuple(updated))


def _best_image(slide: Slide, images: list[DocumentImage]) -> DocumentImage | None:
    """Look to find figure references in text for selecting the best images

    Args:
        slide (Slide): slide
        images (list[DocumentImage]): list of images

    Returns:
        DocumentImage | None: Optimal image for that slide
    """
    slide_words = _tokens(" ".join((slide.title, *slide.bullets)))
    best: tuple[int, DocumentImage] | None = None
    for image in images:
        image_words = _tokens(f"{image.alt_text} {image.path.stem}")
        score = len(slide_words & image_words)
        if score and (best is None or score > best[0]):
            best = (score, image)
    return (
        best[1]
        if best
        else images[0] if not any(slide.image_path for slide in []) else None
    )


def _tokens(text: str) -> set[str]:
    """Set of words and tokens

    Args:
        text (str): text from document

    Returns:
        set[str]: set of tokens for matching
    """
    return {word for word in re.findall(r"[a-zA-Z]{4,}", text.lower())}
