import cv2
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# =========================
# USER SETTINGS
# =========================
output_dir = r"C:\Users\zsong\Desktop\Tow Tank Testing\Ar+Ch Uco Markers"

dictionary_type = cv2.aruco.DICT_4X4_50

fixed_marker_ids = [0, 1, 2]
moving_marker_id = 3

marker_size_mm = 40
border_mm = 5
total_size_mm = marker_size_mm + 2 * border_mm

px_per_mm_fixed = 10
px_per_mm_moving = 5

pdf_filename = os.path.join(output_dir, "aruco_markers_4_letter.pdf")

cut_border_linewidth_mm = 0.25

# =========================
# GENERATE MARKERS
# =========================
os.makedirs(output_dir, exist_ok=True)

dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
saved_files = []

for marker_id in fixed_marker_ids + [moving_marker_id]:

    px_per_mm = (
        px_per_mm_moving if marker_id == moving_marker_id
        else px_per_mm_fixed
    )

    marker_size_px = marker_size_mm * px_per_mm
    border_px = border_mm * px_per_mm

    marker_img = cv2.aruco.generateImageMarker(
        dictionary, marker_id, marker_size_px
    )

    marker_with_border = cv2.copyMakeBorder(
        marker_img,
        border_px, border_px, border_px, border_px,
        borderType=cv2.BORDER_CONSTANT,
        value=255
    )

    filename = os.path.join(output_dir, f"aruco_{marker_id}.png")
    cv2.imwrite(filename, marker_with_border)

    saved_files.append((marker_id, filename))

# =========================
# CREATE PDF - US LETTER
# =========================
c = canvas.Canvas(pdf_filename, pagesize=letter)
page_width, page_height = letter

spacing_x = 20 * mm
spacing_y = 25 * mm
label_gap = 5 * mm

# Center 2 x 2 arrangement
grid_width = 2 * total_size_mm * mm + spacing_x
grid_height = 2 * total_size_mm * mm + spacing_y

start_x = (page_width - grid_width) / 2
start_y = (page_height - grid_height) / 2

positions = [
    (start_x, start_y + total_size_mm * mm + spacing_y),
    (start_x + total_size_mm * mm + spacing_x,
     start_y + total_size_mm * mm + spacing_y),
    (start_x, start_y),
    (start_x + total_size_mm * mm + spacing_x, start_y)
]

for (marker_id, filepath), (x, y) in zip(saved_files, positions):

    label = (
        "Moving marker" if marker_id == moving_marker_id
        else "Fixed reference marker"
    )

    c.setFont("Helvetica", 10)
    c.drawString(
        x,
        y + total_size_mm * mm + label_gap,
        f"ID {marker_id} - {label}"
    )

    c.drawImage(
        filepath,
        x, y,
        width=total_size_mm * mm,
        height=total_size_mm * mm
    )

    # Thin outer cutting border
    c.setLineWidth(cut_border_linewidth_mm * mm)
    c.rect(
        x, y,
        total_size_mm * mm,
        total_size_mm * mm
    )

c.save()

print(f"Saved PDF: {pdf_filename}")