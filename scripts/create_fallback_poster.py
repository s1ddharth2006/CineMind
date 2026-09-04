"""
Generate a sleek Netflix-style fallback poster for CineMind.
"""
from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs("assets", exist_ok=True)
width, height = 500, 750
img = Image.new("RGBA", (width, height), (20, 20, 20, 255))
draw = ImageDraw.Draw(img)

# Dark gradient background
for y in range(height):
    r = int(18 + (30 - 18) * (y / height))
    g = int(18 + (20 - 18) * (y / height))
    b = int(22 + (35 - 22) * (y / height))
    draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

# Subtle border
draw.rectangle([10, 10, width - 10, height - 10], outline=(45, 45, 55, 255), width=2)
# Red accent corner lines
draw.line([(10, 10), (50, 10)], fill=(229, 9, 20, 255), width=3)
draw.line([(10, 10), (10, 50)], fill=(229, 9, 20, 255), width=3)
draw.line([(width - 50, 10), (width - 10, 10)], fill=(229, 9, 20, 255), width=3)
draw.line([(width - 10, 10), (width - 10, 50)], fill=(229, 9, 20, 255), width=3)
draw.line([(10, height - 50), (10, height - 10)], fill=(229, 9, 20, 255), width=3)
draw.line([(10, height - 10), (50, height - 10)], fill=(229, 9, 20, 255), width=3)
draw.line([(width - 50, height - 10), (width - 10, height - 10)], fill=(229, 9, 20, 255), width=3)
draw.line([(width - 10, height - 50), (width - 10, height - 10)], fill=(229, 9, 20, 255), width=3)

# Film clapperboard / reel graphic in the center
cx, cy = width // 2, height // 2 - 40
box_w, box_h = 140, 100
draw.rounded_rectangle([cx - box_w // 2, cy - box_h // 2, cx + box_w // 2, cy + box_h // 2], radius=8, outline=(229, 9, 20, 200), width=3, fill=(28, 28, 35, 255))

# Play icon inside clapperboard
tri_pts = [(cx - 15, cy - 25), (cx - 15, cy + 25), (cx + 25, cy)]
draw.polygon(tri_pts, fill=(229, 9, 20, 255))

# Typography
try:
    font_lg = ImageFont.truetype("arial.ttf", 26)
    font_sm = ImageFont.truetype("arial.ttf", 16)
except Exception:
    font_lg = ImageFont.load_default()
    font_sm = ImageFont.load_default()

text1 = "CINEMIND"
bbox1 = draw.textbbox((0, 0), text1, font=font_lg)
tw1 = bbox1[2] - bbox1[0]
draw.text((cx - tw1 // 2, cy + 80), text1, fill=(255, 255, 255, 255), font=font_lg)

text2 = "Poster Unavailable"
bbox2 = draw.textbbox((0, 0), text2, font=font_sm)
tw2 = bbox2[2] - bbox2[0]
draw.text((cx - tw2 // 2, cy + 120), text2, fill=(150, 150, 160, 255), font=font_sm)

output_path = os.path.join("assets", "fallback_poster.png")
img.save(output_path, "PNG")
print(f"Saved fallback poster to {output_path}")
