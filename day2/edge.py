import cv2
import matplotlib.pyplot as plt
img = cv2.imread("image.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,50,100)
plt.imshow(cv2.cvtColor(edges,cv2.COLOR_BGR2RGB))
plt.show()