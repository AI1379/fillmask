import pytesseract
from PIL import Image
import re


def getText(img: str):
    pytesseract.pytesseract.tesseract_cmd = "tesseract"
    text = pytesseract.image_to_string(Image.open(img))
    lines = [l.strip() for l in text.split("\n") if l]
    txt = []
    que = []
    for i in range(len(lines)):
        if re.match(r"^\(\d{1,2}\)", lines[i]):
            que.append(lines[i])
        else:
            lines[i] = re.sub(r"_*\s*\(\d{1,2}\)\s*_*", " [MASK] ", lines[i])
            lines[i] = re.sub(r"\((.*)\)", "", lines[i])
            lines[i] = re.sub(r"_+", " [MASK] ", lines[i])
            lines[i] = re.sub(r"\|", "I", lines[i])
            txt.append(lines[i])
    return txt, que
