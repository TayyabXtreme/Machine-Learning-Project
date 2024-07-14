import cv2
import numpy as np

            # images


# img=cv2.imread("./mikasa.jpeg")

# cv2.imshow("Mikasa",img)
# cv2.waitKey(0)


            # Video

# cap=cv2.VideoCapture("./test.mp4")
# while True:
#     success,img=cap.read()
#     cv2.imshow("Video",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


            # web came

# cam=cv2.VideoCapture(0)
# cam.set(3,640)
# cam.set(4,480)
# cam.set(10,100)
# while True:
#     success,img=cam.read()
#     cv2.imshow("Video",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


            # image convert into gray scale image and 
            # blur image and canny image and dilation 
            # image and eroded image.



# img=cv2.imread("mikasa.jpeg")
# kernal=np.ones((5,5),np.uint8)


# imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgBlur=cv2.GaussianBlur(imgGray,(7,7),0)
# imgCanny=cv2.Canny(img,100,100)
# imgDilation=cv2.dilate(imgCanny,kernal,iterations=1)
# imgEroded=cv2.erode(imgDilation,kernal,iterations=1)


# cv2.imshow("Canny Image",imgCanny)
# cv2.imshow("Blur Image",imgBlur)
# cv2.imshow("Gray Image",imgGray)
# cv2.imshow("Dilation Image",imgDilation)
# cv2.imshow("Eroded Image",imgEroded)
# cv2.waitKey(0)

            # sizeing and croping image


# img=cv2.imread("mikasa.jpeg")
# print(img.shape)
# imgResize=cv2.resize(img,(500,500))
# imgCropped=img[0:100,200:500]
# cv2.imshow("Cropped Image",imgCropped)

# cv2.imshow("Image",imgResize)
# cv2.waitKey(0)


            # shapes and text

# img=np.zeros((512,512,3),np.uint8)

# img[:]=255,0,0

# cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(255,255,0),3)
# cv2.rectangle(img,(0,0),(250,350),(0,0,255),cv2.FILLED)
# cv2.circle(img,(400,50),30,(255,100,255),cv2.FILLED)
# cv2.putText(img,"OPENCV",(300,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),4)

# cv2.imshow("Image",img)
# cv2.waitKey(0)


                # warp persperctive

# img = cv2.imread("cards2.jpg")

# width,height=250,350

# pts1=np.float32([[111,219],[287,188],[154,482],[352,440]])
# pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
# matrix=cv2.getPerspectiveTransform(pts1,pts2)
# imgOutput=cv2.warpPerspective(img,matrix,(300,300))

# cv2.imshow("Output",imgOutput)

# cv2.imshow("Image",img)
# cv2.waitKey(0)

                    
                    # Joingin images

# def stackImages(scale,imgArray):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range ( 0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor= np.hstack(imgArray)
#         ver = hor
#     return ver

# img = cv2.imread('cards.jpeg')
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# imgStack = stackImages(0.5,([img,imgGray,img],[img,img,img]))

# imgHor = np.hstack((img,img))
# imgVer = np.vstack((img,img))
#
# cv2.imshow("Horizontal",imgHor)
# cv2.imshow("Vertical",imgVer)
# cv2.imshow("ImageStack",imgStack)

# cv2.waitKey(0)


                    # color detection



#                   color detection


# def empty(a):
#     pass

# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars",640,240)
# cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
# cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)
# cv2.createTrackbar("Sat Min","TrackBars",0,255,empty)
# cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
# cv2.createTrackbar("Val Min","TrackBars",0,255,empty)
# cv2.createTrackbar("Val Max","TrackBars",0,255,empty)


# while True:
#     img=cv2.resize(cv2.imread("mikasa.jpeg"),(500,500))
#     imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#     h_min=cv2.getTrackbarPos("Hue Min","TrackBars")
#     h_max=cv2.getTrackbarPos("Hue Max","TrackBars")
#     s_min=cv2.getTrackbarPos("Sat Min","TrackBars")
#     s_max=cv2.getTrackbarPos("Sat Max","TrackBars")
#     v_min=cv2.getTrackbarPos("Val Min","TrackBars")
#     v_max=cv2.getTrackbarPos("Val Max","TrackBars")
#     print(h_min,h_max,s_min,s_max,v_min,v_max)
#     lower=np.array([h_min,s_min,v_min])
#     upper=np.array([h_max,s_max,v_max])
#     mask=cv2.inRange(imgHSV,lower,upper)
#     imgResult=cv2.bitwise_and(img,img,mask=mask)
#     cv2.imshow("Result",imgResult)
#     cv2.imshow("HSV Image",imgHSV)
#     cv2.waitKey(1)




# cv2.imshow("Image",img)


#                       Contours/Shape Detaction

# def stackImages(scale,imgArray):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range ( 0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor= np.hstack(imgArray)
#         ver = hor
#     return ver


# def getContours(img):

#     contours,hiearchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         area=cv2.contourArea(cnt)
        
#         if area>500:
#             cv2.drawContours(imgContour,cnt,-1,(255,0,0),3)
#             peri=cv2.arcLength(cnt,True)
#             # print(peri)
#             approx=cv2.approxPolyDP(cnt,0.02*peri,True)
#             print(len(approx))
#             objCor=len(approx)
#             x,y,w,h=cv2.boundingRect(approx)
            
#             if objCor==3: objectType="Tri"
#             elif objCor==4:
#                 aspRatio=w/float(h)
#                 if aspRatio>0.95 and aspRatio<1.05: objectType="Square"
#                 else: objectType="Rectangle"
#             elif objCor==5: objectType="diamond"
#             elif objCor>6: objectType="Circle"
#             else: objectType="None"

#             cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
#             cv2.putText(imgContour,objectType,(x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),2)


            



                                    



# img=cv2.imread("shapes.jpg")
# imgContour=img.copy()
# imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgBlur=cv2.GaussianBlur(imgGray,(7,7),1)
# imgCanny=cv2.Canny(imgBlur,50,50)
# getContours(imgCanny)
# imgBlack=np.zeros_like(img)

# imgStack=stackImages(0.6,([img,imgGray,imgBlur],[imgCanny,imgContour,imgBlur]))





# cv2.imshow("Stack",imgStack)

# cv2.waitKey(0)









                    #   Face Detection    

faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img=cv2.imread("mikasa.jpeg")
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(imgGray,1.1,4)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)





cv2.imshow("Result",img)
cv2.waitKey(0)

