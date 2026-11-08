
import cv2
import os
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# =========================
# USER SETTINGS
# =========================
output_dir = r"C:\Users\zsong\Desktop\Tow Tank Testing\Ar+Ch Uco Markers"
os.makedirs(output_dir, exist_ok=True)

dictionary_type = cv2.aruco.DICT_4X4_250

squares_x = 16
squares_y = 10
square_size_mm = 25
marker_size_mm = 18

board_width_mm = squares_x * square_size_mm
board_height_mm = squares_y * square_size_mm

px_per_mm = 10

png_filename = os.path.join(output_dir, "charuco_board.png")
pdf_filename = os.path.join(output_dir, "charuco_board.pdf")
split_pdf_filename = os.path.join(output_dir, "charuco_board_split.pdf")

# =========================
# GENERATE CHARUCO BOARD
# =========================
dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)

board = cv2.aruco.CharucoBoard(
    (squares_x, squares_y),
    square_size_mm,
    marker_size_mm,
    dictionary
)

board_image = board.generateImage(
    (
        int(board_width_mm * px_per_mm),
        int(board_height_mm * px_per_mm)
    ),
    marginSize=0
)

cv2.imwrite(png_filename, board_image)

# =========================
# CREATE FULL PDF - 11 x 17
# =========================
page_width = 17 * 25.4 * mm
page_height = 11 * 25.4 * mm

c = canvas.Canvas(
    pdf_filename,
    pagesize=(page_width, page_height)
)

x = (page_width - board_width_mm * mm) / 2
y = (page_height - board_height_mm * mm) / 2

c.drawImage(
    png_filename,
    x, y,
    width=board_width_mm * mm,
    height=board_height_mm * mm
)

c.save()

# =========================
# SPLIT INTO TWO LETTER PAGES
# =========================
reader = PdfReader(pdf_filename)
writer = PdfWriter()

source_page = reader.pages[0]

letter_width = 8.5 * 25.4 * mm
letter_height = 11 * 25.4 * mm

half_width = board_width_mm * mm / 2
margin_x = (letter_width - half_width) / 2

for i in range(2):
    page = writer.add_blank_page(
        width=letter_width,
        height=letter_height
    )

    # Move each half of the full pattern onto its own page
    offset_x = margin_x - (x + i * half_width)

    page.merge_transformed_page(
        source_page,
        Transformation().translate(tx=offset_x, ty=0)
    )

with open(split_pdf_filename, "wb") as f:
    writer.write(f)

print(f"Saved PNG: {png_filename}")
print(f"Saved full PDF: {pdf_filename}")
print(f"Saved split PDF: {split_pdf_filename}")
