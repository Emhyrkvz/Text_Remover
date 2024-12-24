import easyocr
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

reader = easyocr.Reader(['en'])
img_path = r'D:\Projects\images\Cars1.png'
img = cv.imread(img_path)

mask = np.zeros(img.shape[:2], dtype=np.uint8)

result = reader.readtext(img_path)

for (bbox, text, prob) in result:
    print(f'Text: {text}, Probability: {prob}')
    top_left = tuple(map(int, bbox[0]))
    bottom_right = tuple(map(int, bbox[2]))
    cv.rectangle(mask, top_left, bottom_right, (255,255,255), -1)

inpainted_img = cv.inpaint(img, mask, inpaintRadius=3, flags=cv.INPAINT_TELEA)

cv.waitKey(0)
cv.destroyAllWindows()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Inpainted Image')
plt.imshow(cv.cvtColor(inpainted_img, cv.COLOR_BGR2RGB))

plt.show()