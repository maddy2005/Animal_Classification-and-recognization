from ultralytics import YOLO  
import cv2
import os

#Load train model 
model = YOLO(r'C:\Users\dspvi\OneDrive\Desktop\Animal_Classification\best.pt')

#Input image path
image_path  = r'C:\Users\dspvi\OneDrive\Desktop\Animal_Classification\train\images\Rhino_14.jpg'

results = model(image_path,save =True,conf=0.25)

#get saved image path 
output_dir = results[0].save_dir
output_img_path = os.path.join(output_dir, os.path.basename(image_path))
#Load and display results
img = cv2.imread(output_img_path)
cv2.imshow('detection',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
