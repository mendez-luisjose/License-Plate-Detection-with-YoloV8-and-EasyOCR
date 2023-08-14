import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
from util import set_background, write_csv
import uuid
import os
from  streamlit_webrtc import webrtc_streamer
import av

set_background("./imgs/background.png")

folder_path = "./licenses_plates_imgs_detected/"
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"

reader = easyocr.Reader(['en'], gpu=False)

vehicles = [2]

header = st.container()
body = st.container()

coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

threshold = 0.15

state = "Uploader"

if "state" not in st.session_state :
    st.session_state["state"] = "Uploader"

class VideoProcessor:
    def recv(self, frame) :
        img = frame.to_ndarray(format="bgr24")
        img_to_an = img.copy()
        img_to_an = cv2.cvtColor(img_to_an, cv2.COLOR_RGB2BGR)
        license_detections = license_plate_detector(img_to_an)[0]

        if len(license_detections.boxes.cls.tolist()) != 0 :
            for license_plate in license_detections.boxes.data.tolist() :
                x1, y1, x2, y2, score, class_id = license_plate

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

                license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
            
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 

                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

                cv2.rectangle(img, (int(x1) - 40, int(y1) - 40), (int(x2) + 40, int(y1)), (255, 255, 255), cv2.FILLED)
                cv2.putText(img,
                            str(license_plate_text),
                            (int((int(x1) + int(x2)) / 2) - 70, int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0),
                            3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    

def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop)

    width = img.shape[1]
    height = img.shape[0]
    
    if detections == [] :
        return None, None

    rectangle_size = license_plate_crop.shape[0]*license_plate_crop.shape[1]

    plate = [] 

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > 0.17:
            bbox, text, score = result
            text = result[1]
            text = text.upper()
            scores += score
            plate.append(text)
    
    if len(plate) != 0 : 
        return " ".join(plate), scores/len(plate)
    else :
        return " ".join(plate), 0

def model_prediction(img):
    license_numbers = 0
    results = {}
    licenses_texts = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    object_detections = coco_model(img)[0]
    license_detections = license_plate_detector(img)[0]

    if len(object_detections.boxes.cls.tolist()) != 0 :
        for detection in object_detections.boxes.data.tolist() :
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection

            if int(class_id) in vehicles :
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
    else :
            xcar1, ycar1, xcar2, ycar2 = 0, 0, 0, 0
            car_score = 0

    if len(license_detections.boxes.cls.tolist()) != 0 :
        license_plate_crops_total = []
        for license_plate in license_detections.boxes.data.tolist() :
            x1, y1, x2, y2, score, class_id = license_plate

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

            img_name = '{}.jpg'.format(uuid.uuid1())
         
            cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)

            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 

            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

            licenses_texts.append(license_plate_text)

            if license_plate_text is not None and license_plate_text_score is not None  :
                license_plate_crops_total.append(license_plate_crop)
                results[license_numbers] = {}
                
                results[license_numbers][license_numbers] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'car_score': car_score},
                                                        'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                            'text': license_plate_text,
                                                                            'bbox_score': score,
                                                                            'text_score': license_plate_text_score}} 
                license_numbers+=1
          
        write_csv(results, f"./csv_detections/detection_results.csv")

        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        return [img_wth_box, licenses_texts, license_plate_crops_total]
    
    else : 
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box]
    

def change_state_uploader() :
    st.session_state["state"] = "Uploader"

    
def change_state_camera() :
    st.session_state["state"] = "Camera"

def change_state_live() :
    st.session_state["state"] = "Live"
    
with header :
    _, col1, _ = st.columns([0.2,1,0.1])
    col1.title("💥 License Car Plate Detection 🚗")

    _, col0, _ = st.columns([0.15,1,0.1])
    col0.image("./imgs/test_background.jpg", width=500)


    _, col4, _ = st.columns([0.1,1,0.2])
    col4.subheader("Computer Vision Detection with YoloV8 🧪")

    _, col, _ = st.columns([0.3,1,0.1])
    col.image("./imgs/plate_test.jpg")

    _, col5, _ = st.columns([0.05,1,0.1])

    st.write("The differents models detect the car and the license plate in a given image, then extracts the info about the license using EasyOCR, and crop and save the license plate as a Image, with a CSV file with all the data.   ")


with body :
    _, col1, _ = st.columns([0.1,1,0.2])
    col1.subheader("Check It-out the License Car Plate Detection Model 🔎!")

    _, colb1, colb2, colb3 = st.columns([0.2, 0.7, 0.6, 1])

    if colb1.button("Upload an Image", on_click=change_state_uploader) :
        pass
    elif colb2.button("Take a Photo", on_click=change_state_camera) :
        pass
    elif colb3.button("Live Detection", on_click=change_state_live) :
        pass

    if st.session_state["state"] == "Uploader" :
        img = st.file_uploader("Upload a Car Image: ", type=["png", "jpg", "jpeg"])
    elif st.session_state["state"] == "Camera" :
        img = st.camera_input("Take a Photo: ")
    elif st.session_state["state"] == "Live" :
        webrtc_streamer(key="sample", video_processor_factory=VideoProcessor)
        img = None

    _, col2, _ = st.columns([0.3,1,0.2])

    _, col5, _ = st.columns([0.8,1,0.2])

    
    if img is not None:
        image = np.array(Image.open(img))    
        col2.image(image, width=400)

        if col5.button("Apply Detection"):
            results = model_prediction(image)

            if len(results) == 3 :
                prediction, texts, license_plate_crop = results[0], results[1], results[2]

                texts = [i for i in texts if i is not None]
                
                if len(texts) == 1 and len(license_plate_crop) :
                    _, col3, _ = st.columns([0.4,1,0.2])
                    col3.header("Detection Results ✅:")

                    _, col4, _ = st.columns([0.1,1,0.1])
                    col4.image(prediction)

                    _, col9, _ = st.columns([0.4,1,0.2])
                    col9.header("License Cropped ✅:")

                    _, col10, _ = st.columns([0.3,1,0.1])
                    col10.image(license_plate_crop[0], width=350)

                    _, col11, _ = st.columns([0.45,1,0.55])
                    col11.success(f"License Number: {texts[0]}")

                    df = pd.read_csv(f"./csv_detections/detection_results.csv")
                    st.dataframe(df)
                elif len(texts) > 1 and len(license_plate_crop) > 1  :
                    _, col3, _ = st.columns([0.4,1,0.2])
                    col3.header("Detection Results ✅:")

                    _, col4, _ = st.columns([0.1,1,0.1])
                    col4.image(prediction)

                    _, col9, _ = st.columns([0.4,1,0.2])
                    col9.header("License Cropped ✅:")

                    _, col10, _ = st.columns([0.3,1,0.1])

                    _, col11, _ = st.columns([0.45,1,0.55])

                    col7, col8 = st.columns([1,1])
                    for i in range(0, len(license_plate_crop)) :
                        col10.image(license_plate_crop[i], width=350)
                        col11.success(f"License Number {i}: {texts[i]}")

                    df = pd.read_csv(f"./csv_detections/detection_results.csv")
                    st.dataframe(df)
            else :
                prediction = results[0]
                _, col3, _ = st.columns([0.4,1,0.2])
                col3.header("Detection Results ✅:")

                _, col4, _ = st.columns([0.3,1,0.1])
                col4.image(prediction)




 