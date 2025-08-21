import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((500,500),dtype=np.uint8)
circle = cv2.circle(img.copy(),(250,250),250,(255,255,255),-1)
rect = cv2.rectangle(img.copy(),(30,30),(470,470),(255,255,255),-1)

intersection = cv2.bitwise_and(circle,rect,None)

plt.imshow(cv2.cvtColor(intersection,cv2.COLOR_BGR2RGB))
plt.show()