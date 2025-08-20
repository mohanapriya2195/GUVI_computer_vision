import cv2
import matplotlib.pyplot as plt
img = cv2.imread("noise.jpg")

blur=cv2.blur(img,(5,5))
plt.imshow(cv2.cvtColor(blur,cv2.COLOR_BGR2RGB))
plt.show()