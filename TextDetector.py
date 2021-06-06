!sudo apt install tesseract-ocr
!pip install pytesseract
try:
    from PIL import Image
except ImportError:
    import Image

import pytesseract
for i in range(2):
  print("\n")
print(pytesseract.image_to_string(Image.open('/content/drive/MyDrive/Untitled1.png')))
