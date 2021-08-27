import cv2

img1 = cv2.imread("C:/Users/tseng/Desktop/projects/NISwGSP/x64/Release/Input-42-data/human/0.png",cv2.IMREAD_UNCHANGED)
img2 = cv2.imread("C:/Users/tseng/Desktop/projects/NISwGSP/x64/Release/Input-42-data/human/1.png",cv2.IMREAD_UNCHANGED)

result = img1.copy()
w,h,c = result.shape

for w in range(0,w):
    for h in range(0,h):
        result[w][h]  = img1[w][h]/2+img2[w][h]/2

cv2.imwrite('blendingResult.jpg',result)