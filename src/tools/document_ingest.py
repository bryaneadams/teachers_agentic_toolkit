from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from errors import InputValidationError
from models import DocumentImage, Document

IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def load_document(
    document_path: str | Path, image_dir: str | Path | None = None
) -> Document:
    """loads the document if in the accepted formats

    Args:
        document_path (str | Path): path to the document
        image_dir (str | Path | None, optional): path to and imagery directory. Defaults to None.

    Raises:
        InputValidationError: File does not exist
        InputValidationError: File not in a valid file format

    Returns:
        Document: The ingested document
    """
    path = Path(document_path)
    if not path.exists():
        raise InputValidationError(f"Document file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        document = _load_text_document(path)
        images = list(document.images)
        text = document.text
        title = document.title
    elif suffix == ".pdf":
        document = _load_pdf_document(path)
        images = list(document.images)
        text = document.text
        title = document.title
    else:
        raise InputValidationError(
            "Document input supports .pdf, .txt, .md, and .markdown files."
        )

    if image_dir is not None:
        images.extend(_images_from_directory(Path(image_dir)))
    return Document(title=title, text=text, images=tuple(dict.fromkeys(images)))


def _load_text_document(path: Path) -> Document:
    """Loads text documents

    Args:
        path (Path): path to the text document

    Raises:
        InputValidationError: If the file is empty

    Returns:
        Document: document text
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise InputValidationError("Document file is empty.")
    return Document(
        title=_title_from_text(text, path.stem),
        text=text,
        images=_images_from_markdown(text, path.parent),
    )


def _load_pdf_document(path: Path) -> Document:
    """Reads in the pdf

    Args:
        path (Path): path to the pdf

    Raises:
        InputValidationError: Error reading the pdf
        InputValidationError: Error extracting text from a page
        InputValidationError: No text extracted from the page

    Returns:
        Document: document of text
    """
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise InputValidationError(f"Could not read PDF document: {path}") from exc

    pages = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise InputValidationError(
                f"Could not extract text from PDF page {index}."
            ) from exc
        if page_text.strip():
            pages.append(page_text.strip())

    text = "\n\n".join(pages).strip()
    if not text:
        raise InputValidationError("PDF document contains no extractable text.")

    return Document(
        title=_pdf_title(reader, text, path.stem),
        text=text,
        images=_extract_pdf_images(reader, path),
    )


def _title_from_text(text: str, fallback: str) -> str:
    """Finds title in the text

    Args:
        text (str): document text
        fallback (str): default string

    Returns:
        str: text title if found
    """
    for line in text.splitlines():
        clean = line.strip()
        if clean.startswith("#"):
            return clean.lstrip("#").strip() or fallback
        if clean:
            return clean[:80]
    return fallback


def _images_from_markdown(text: str, base_dir: Path) -> tuple[DocumentImage, ...]:
    """Finds images in markdown files

    Args:
        text (str): document text
        base_dir (Path): path to save images

    Returns:
        tuple[DocumentImage, ...]: tuple of extracted images
    """
    images: list[DocumentImage] = []
    for match in IMAGE_RE.finditer(text):
        raw_path = match.group("path").strip()
        if raw_path.startswith(("http://", "https://")):
            continue
        image_path = (base_dir / raw_path).resolve()
        if image_path.exists():
            images.append(
                DocumentImage(
                    path=image_path,
                    alt_text=match.group("alt").strip(),
                    source="markdown",
                )
            )
    return tuple(images)


def _images_from_directory(image_dir: Path) -> tuple[DocumentImage, ...]:
    """Pulls images from a directory

    Args:
        image_dir (Path): path to image directory

    Raises:
        InputValidationError: raises error if your image directory does not exist

    Returns:
        tuple[DocumentImage, ...]: tuple of images from your directory
    """
    if not image_dir.exists():
        raise InputValidationError(f"Image directory does not exist: {image_dir}")
    return tuple(
        DocumentImage(
            path=path.resolve(),
            alt_text=path.stem.replace("_", " "),
            source="image_dir",
        )
        for path in sorted(image_dir.iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()
    )


def _pdf_title(reader, text: str, fallback: str) -> str:
    """Extracts title from a pdf

    Args:
        reader (_type_): your pdf reader
        text (str): document text
        fallback (str): default string

    Returns:
        str: extracted title
    """
    metadata_title = getattr(getattr(reader, "metadata", None), "title", None)
    if metadata_title:
        return str(metadata_title).strip()[:120]
    return _title_from_text(text, fallback)


def _extract_pdf_images(reader, pdf_path: Path) -> tuple[DocumentImage, ...]:
    """Finds images in a pdf

    Args:
        reader (_type_): pdf reader
        pdf_path (Path): path to pdf

    Returns:
        tuple[DocumentImage, ...]: tuple of extracted images
    """
    output_dir = pdf_path.parent / f"{pdf_path.stem}_images"
    images: list[DocumentImage] = []
    for page_number, page in enumerate(reader.pages, start=1):
        for image_number, image_file_object in enumerate(_page_images(page), start=1):
            name = _pdf_image_name(
                pdf_path.stem, page_number, image_number, image_file_object
            )
            image_path = output_dir / name
            try:
                data = image_file_object.data
            except Exception:
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(data)
            images.append(
                DocumentImage(
                    path=image_path.resolve(),
                    alt_text=f"{pdf_path.stem} page {page_number} image {image_number}",
                    source="pdf",
                )
            )
    return tuple(images)


def _page_images(page) -> Iterable:
    """_summary_

    Args:
        page (_type_): _description_

    Returns:
        Iterable: _description_
    """
    try:
        return page.images
    except Exception:
        return ()


def _pdf_image_name(
    pdf_stem: str, page_number: int, image_number: int, image_file_object
) -> str:
    """finds name of image if possible

    Args:
        pdf_stem (str): pdf stem
        page_number (int): page number
        image_number (int): image number
        image_file_object (_type_): type of image

    Returns:
        str: name of image
    """
    raw_name = getattr(image_file_object, "name", "") or ""
    suffix = Path(raw_name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".png"
    return f"{pdf_stem}_page_{page_number}_image_{image_number}{suffix}"
