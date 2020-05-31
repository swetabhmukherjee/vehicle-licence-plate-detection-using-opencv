import cv2
import matplotlib.pyplot as plt
import imutils


img=cv2.imread('car6.jpeg') 
cv2.imshow('image',img) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 
plt.imshow(img)


mg = cv2.resize(img, (620,480) )
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale 
cv2.imshow('gray image',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


#bilateral filtering
gray = cv2.bilateralFilter(gray, 11, 17, 17) 
cv2.imshow('bilaeral image',gray) 
cv2.waitKey(0)
cv2.destroyAllWindows()


#edge detection
edged = cv2.Canny(gray, 30, 200) #Perform Edge detection 
cv2.imshow('edge detection',edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None


# loop over our contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.018 * peri, True)
	# if our approximated contour has four points, then # we can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx 
		break


import numpy as np
# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,) 
new_image = cv2.bitwise_and(img,img,mask=mask)


# Now crop
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y)) 
(bottomx, bottomy) = (np.max(x), np.max(y)) 
Cropped = gray[topx:bottomx+1, topy:bottomy+1]


#show number plate image
cv2.imshow('img',new_image) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 
plt.imshow(new_image)


#show cropped image
cv2.imshow('img',Cropped) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 
plt.imshow(Cropped) 
cv2.imwrite('plate.jpg',img)