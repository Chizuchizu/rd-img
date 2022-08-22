import numpy as np
import cv2


array = np.random.randint(0, 256, (10, 10, 3))
array2 = np.random.randint(0, 256, (10, 80, 3))
array2 = np.zeros((10, 80, 3)).astype(np.uint8) + 255

print(array)

cv2.imwrite("1010.png", array)
cv2.imwrite("1080.png", array2)
