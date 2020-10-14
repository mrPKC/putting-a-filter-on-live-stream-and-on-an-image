from cv2 import cv2
import numpy as np 
import pandas as pd 

cap = cv2.VideoCapture(0)
glass = cv2.imread("snapchat project\\glasses.png",-1)
gorig_mask = glass[:,:,3]
gorig_mask_inv = cv2.bitwise_not(gorig_mask)
glass = glass[:,:,0:3]
gorigHeight, gorigWidth = glass.shape[:2]

mucch = cv2.imread("snapchat project\\mustache.png",-1)
orig_mask = mucch[:,:,3]
orig_mask_inv = cv2.bitwise_not(orig_mask)
mucch = mucch[:,:,0:3]
origMustacheHeight, origMustacheWidth = mucch.shape[:2]

eye_cascade = cv2.CascadeClassifier("haar_cascade\\haarcascade_eye.xml")
nose_cascade = cv2.CascadeClassifier("haar_cascade\\haarcascade_nose.xml")
face_cascade = cv2.CascadeClassifier("haar_cascade\\haarcascade_frontalface_alt.xml")
# img = cv2.imread("snapchat project\\Before.png")
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cords = []


while(True):
    ret,frame = cap.read()

    if(ret == False):
        continue
    
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame,1.3,5)
    
    for (x,y,w,h) in faces:
  #  cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray_frame[y:y+h, x:x+h]
        roi_color = frame[y:y+h, x:x+h]
        muchhs = nose_cascade.detectMultiScale(roi_gray)
        for (a,b,c,d) in muchhs:
            # #cv2.rectangle(roi_color,(a-c,b+d-10),(a+c+c,b+d+15),(255,0,0),2)

            mustacheWidth =  int(c)
            mustacheHeight = int((1.6*mustacheWidth) * origMustacheHeight / origMustacheWidth)

            x1 = int(a - (mustacheWidth/4))
            x2 = int(a + c + (mustacheWidth/4))
            y1 = int(b + d - (mustacheHeight/2))
            y2 = int(b + d + (mustacheHeight/2))

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h

            mustacheWidth = int(x2 - x1)
            mustacheHeight = int(y2 - y1)

            mucch = cv2.resize(mucch, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)

            roi = roi_color[y1:y2, x1:x2]

            roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

            roi_fg = cv2.bitwise_and(mucch,mucch,mask = mask)

            dst = cv2.add(roi_bg,roi_fg)

            roi_color[y1:y2, x1:x2] = dst

        eyes = eye_cascade.detectMultiScale(roi_gray,1.5,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #         cords.append([ex,ey])
        
    #     glass_w = int(cords[0][0]+ew - cords[1][0])
    #     glass_h = int(glass_w * gorigHeight/gorigWidth)

    #     exn = int(cords[1][0])
    #     exnf = int(cords[1][0] + glass_w)
    #     eyn = int(cords[1][1])
    #     eynf = int(cords[1][1] + glass_h)

    #     if exn < 0:
    #         exn = 0
    #     if eyn < 0:
    #         eyn = 0
    #     if exnf > w:
    #         exnf = w
    #     if eynf > h:
    #         eynf = h

    #     glass_w = int(exnf - exn)
    #     glass_h = int(eynf - eyn)

    #     glass = cv2.resize(glass, (glass_w,glass_h), interpolation=cv2.INTER_AREA)
    #     gmask = cv2.resize(gorig_mask, (glass_w,glass_h), interpolation=cv2.INTER_AREA)
    #     gmask_inv = cv2.resize(gorig_mask_inv, (glass_w,glass_h), interpolation=cv2.INTER_AREA)

    #     roig = roi_color[eyn:eynf,exn:exnf]

    #     groi_bg = cv2.bitwise_and(roig,roig,mask = gmask_inv)

    #     groi_fg = cv2.bitwise_and(glass,glass,mask = gmask)

    #     dstg = cv2.add(groi_bg,groi_fg)

    #     roi_color[eyn:eynf,exn:exnf] = dstg

    #    # print(len(cords))
    #     cords = []
    cv2.imshow("filter",frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if(key_pressed == ord("s")):
        break

cap.release()
cv2.destroyAllWindows()