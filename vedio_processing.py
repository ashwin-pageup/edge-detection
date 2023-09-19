import cv2
import os
import numpy as np
import pandas as pd
import pypyodbc as db

driver = 'SQL Server'
server = 'ASUS\SQLEXPRESS'
database = 'coordinates'
user = 'sa'
password = 'pageupdev'

conn_str = f"""
    DRIVER={{{driver}}};
    SERVER={server};
    DATABASE={database};
    UID={user};
    PWD={password};
"""

conn = db.connect(conn_str)

cursor = conn.cursor()

cap = cv2.VideoCapture('cars.mp4')
curr_frame = 0

if not os.path.exists('images'):
    os.makedirs('images')

j=0
while cap.isOpened():
    j+=1
    success, frame = cap.read()
    if not success:
        break

    lower_bound = np.array([0, 0, 170])
    upper_bound = np.array([179, 255, 255])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tree_contours = []
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Calculate the aspect ratio of the contour
        aspect_ratio = cv2.boundingRect(contour)[2] / cv2.boundingRect(contour)[3]

        if (area > 90000 or area < 40000) and 1 < aspect_ratio < 2:
            tree_contours.append(contour)

    #clearing sky
    for contour in tree_contours:
        cv2.fillPoly(frame, [contour], (0, 0, 0))


    # white car detection : 

    # Convert the image to HSV color space - hue-color, intensity, brightness
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the white color range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply the mask to the image to remove everything from the background
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    #cleaning small patterns
    gray = cv2.cvtColor(foreground,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if(cv2.contourArea(contour)<100):
            cv2.fillPoly(thresh, [contour], (0, 0, 0))
    
    # EDGE DETECTION
    edge = cv2.Canny(thresh, 70, 140)

    # writing to database : SSMS
    for i in range(len(edge)):
        for j in range(len(edge[i])):
            if(edge[i][j]==255):
                cursor.execute("INSERT INTO Coordinates VALUES ('frame"+str(curr_frame)+"',"+str(i)+","+str(j)+");")
                conn.commit()

    # creating img
    cv2.imwrite('./images/'+str(curr_frame)+'.jpg',edge)
    curr_frame += 1


# code for creating the vedio
path = "./images/"
img_list = os.listdir("./images")
out = "cars_edge1.mp4"
size = list((cv2.imread("./images/0.jpg")).shape)
del(size[2])
size.reverse()
video = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

for i in range(len(img_list)):
    video.write(cv2.imread('./images/'+str(i)+'.jpg'))
video.release()
    
