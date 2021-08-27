import cv2
import os 
# Opens the Video file
outputdir = './20210820-146frames-testing-data_CDCL/'
cap= cv2.VideoCapture(os.path.join(outputdir,'output.avi'))
out = None
while(cap.isOpened()):
    ret, videoframe = cap.read()
    if ret == False:
        break

    if out is None:
        fps = 15
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(outputdir,'output-2.avi'), fourcc, fps , (videoframe.shape[1],  videoframe.shape[0]))

    out.write(videoframe)


cap.release()
out.release()
