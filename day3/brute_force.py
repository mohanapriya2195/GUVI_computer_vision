import cv2
img1 = cv2.imread('image2.jpg')
img2 = cv2.imread('image3.jpg')
orb=cv2.ORB_create()
kp1,des1=orb 