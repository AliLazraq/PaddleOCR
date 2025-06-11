import os
import sys

# Add CUDNN to PATH
cudnn_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cudnn', 'bin')
print(f"CUDNN path: {cudnn_path}")
print(f"CUDNN exists: {os.path.exists(cudnn_path)}")

if os.path.exists(cudnn_path):
    os.environ['PATH'] = cudnn_path + ';' + os.environ.get('PATH', '')
    print("CUDNN added to PATH")

# Also try cublas path
cublas_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cublas', 'bin')
if os.path.exists(cublas_path):
    os.environ['PATH'] = cublas_path + ';' + os.environ.get('PATH', '')
    print("CUBLAS added to PATH")

import paddle
print("CUDA available:", paddle.device.is_compiled_with_cuda())

from paddleocr import PaddleOCR
import fitz

# Test GPU
ocr = PaddleOCR(lang='en')

doc = fitz.open('data/Ali_Lazraq.pdf')
page = doc[0]
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
img_data = pix.tobytes("png")

with open('temp_page.png', 'wb') as f:
    f.write(img_data)

print("Testing GPU OCR...")
result = ocr.ocr('temp_page.png')

if result and result[0]:
    print("SUCCESS! GPU is working:")
    for line in result[0]:
        print(f"Text: {line[1][0]}, Confidence: {line[1][1]:.2f}")
else:
    print("No text found")

doc.close()