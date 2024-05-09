from pathlib import Path
from typing import List, Union
import logging

from cloudpathlib import CloudPath
import fitz


# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


def get_pdf_quadrants(num_of_quadrants: int, page_rect: fitz.Rect) -> List[fitz.Rect]:
    if num_of_quadrants in [1, 2, 4, 6, 8, 10]:
        if num_of_quadrants == 1:
            rows, cols = 1, 1
        elif num_of_quadrants == 2:
            rows, cols = 1, 2
        elif num_of_quadrants == 4:
            rows, cols = 2, 2
        elif num_of_quadrants == 6:
            rows, cols = 2, 3
        elif num_of_quadrants == 8:
            rows, cols = 2, 4
        elif num_of_quadrants == 10:
            rows, cols = 2, 5
    else:
        raise ValueError(
            "Unsupported number of quadrants. Supported values are 1, 2, 4, 6, 8, and 10."
        )

    quadrant_width = page_rect.width / cols
    quadrant_height = page_rect.height / rows
    quadrants = []
    for i in range(rows):
        for j in range(cols):
            x0 = page_rect.x0 + j * quadrant_width
            y0 = page_rect.y0 + i * quadrant_height
            x1 = x0 + quadrant_width
            y1 = y0 + quadrant_height
            quadrants.append(fitz.Rect(x0, y0, x1, y1))
    return quadrants


def create_nup_pdf(
    filepath: Union[str, Path, CloudPath],
    nup_pages: int,
    output_filepath: Union[str, Path, CloudPath],
    page_size: str = "letter",
):
    src = fitz.open(filepath)

    width, height = fitz.paper_size(page_size)
    page_rectangle = fitz.Rect(0, 0, width, height)
    quadrants = get_pdf_quadrants(nup_pages, page_rectangle)

    # now copy input pages to output
    doc = fitz.open()
    skipped_pages = 0
    for src_page in src:
        if src_page.get_contents() == []:
            logger.info(f"Page {src_page.number} is empty. Skipping.")
            skipped_pages += 1
            continue
        if (src_page.number - skipped_pages) % nup_pages == 0:  # create new output page
            page = doc.new_page(-1, width=width, height=height)
        # insert input page into the correct rectangle
        page.show_pdf_page(
            quadrants[
                (src_page.number - skipped_pages) % nup_pages
            ],  # select output rect
            src,  # input document
            src_page.number,
        )  # input page number

    # by all means, save new file using garbage collection and compression
    doc.save(output_filepath, garbage=4, deflate=True)
