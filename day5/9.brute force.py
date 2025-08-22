import cv2
import matplotlib.pyplot as plt
img1 = cv2.imread('image2.jpg')
img2 = cv2.imread('image3.jpg')

#ORB Keypoint Extractor
orb= cv2.ORB_create()
kp1,des1=orb.detectAndCompute(img1,None)
kp2,des2=orb.detectAndCompute(img2,None)

Brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = Brute_force.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30],None,flags=2)

plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2GRAY))
plt.show()