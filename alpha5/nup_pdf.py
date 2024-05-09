from typing import List

import fitz


def get_pdf_quadrants(num_of_quadrants: int, page_rect: fitz.Rect) -> List[fitz.Rect]:
    rows = cols = int(num_of_quadrants**0.5)
    if num_of_quadrants == 2:
        cols = 2  # For 2 quadrants, we need 2 columns and 1 row
        rows = 1
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
    filepath: str,
    nup_pages: int,
    output_filepath: str,
    page_size: str = "letter",
):
    src = fitz.open(filepath)

    width, height = fitz.paper_size(page_size)
    page_rectangle = fitz.Rect(0, 0, width, height)
    quadrants = get_pdf_quadrants(nup_pages, page_rectangle)

    # now copy input pages to output
    doc = fitz.open()
    for src_page in src:
        if src_page.number % nup_pages == 0:  # create new output page
            page = doc.new_page(-1, width=width, height=height)
        # insert input page into the correct rectangle
        print(quadrants)
        print(src_page.number % nup_pages)
        page.show_pdf_page(
            quadrants[src_page.number % nup_pages],  # select output rect
            src,  # input document
            src_page.number,
        )  # input page number

    # by all means, save new file using garbage collection and compression
    doc.save(output_filepath, garbage=4, deflate=True)
