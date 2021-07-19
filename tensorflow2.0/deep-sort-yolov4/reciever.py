#! /usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division, print_function, absolute_import
import os
from timeit import time
import warnings
import cv2

import numpy as np
from PIL import Image
#from yolo import YOLO
import random

import zmq
import pickle
import zlib

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection

from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
import matching
from videocaptureasync import VideoCaptureAsync

import panorama
warnings.filterwarnings('ignore')



import imagezmq

#image_hub = imagezmq.ImageHub()

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")	
socket.connect("tcp://localhost:5555")


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



import logging# Starting a logger for the server

root_logger = logging.getLogger('Logger')
root_logger.setLevel(logging.DEBUG)

handler = logging.FileHandler('logger.txt')
handler.setLevel(logging.DEBUG)
log_format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
handler.setFormatter(log_format)
root_logger.addHandler(handler)



CUDA_VISIBLE_DEVICES=""

#def main(yolo):
def main():
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    two_cameras = False
    same_view = False

    recieving = True
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]
    
    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    tracking = True
    writeVideo_flag = True
    asyncVideo_flag = False

    file_path = '/home/ata/Desktop/football/football_left.mp4'
    if asyncVideo_flag :
        video_capture = VideoCaptureAsync(file_path)
    else:
        video_capture = cv2.VideoCapture(file_path)

    if asyncVideo_flag:
        video_capture.start()
    
    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
        else:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        if two_cameras is False:
            
            h = 800
            w = 800
            #fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter('output_yolov4.avi', fourcc, 30, (w, h))
            frame_index = -1
    

    
    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
    frame_index = 0
    if two_cameras is True:

        cap2 = cv2.VideoCapture('/home/ata/Desktop/football/football_right.mp4')
    flag = True
    while True:
        
        if not recieving:
            ret, frame = video_capture.read()  # frame shape 640*480*3
        else:
            h = 800
            w = 800	
            #rpi_name, frame = image_hub.recv_image()
            #image_hub.send_reply(b'OK')
            #background = Image.new('RGBA', (w, h), (255, 255, 255, 255))
            background = np.zeros((h, w, 3)).astype(np.uint8)
          
            #background = Image.new('RGB', (480, 640))
            z = socket.recv(flags=0)
            p = zlib.decompress(z)
            detections = pickle.loads(p)
            i = 0
            boxes = []
            confidence = []
            classes = []
            for p in detections:
                bbox = [int(i) for i in p[1]]
                print(bbox)
                boxes.append(bbox)
                confidence.append(80)
                classes.append(int(p[2]))
                #p[0] = Image.fromarray(p[0][...,::-1]) 
                
                
                #cv2.imshow('', p[0])
                i += 1
                #print(bbox)
                
                background[int(bbox[1]):int(bbox[1]) +int(bbox[3]) ,int(bbox[0]) : int(bbox[0]) + int(bbox[2]), :] = p[0]
                
                #background = Image.fromarray(background[...,::-1].astype(np.uint8))
                #background.paste(p[0], (10, 20, 30, 40))
            #cv2.imshow('opx', background)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            #continue
            
        ret = True
        if ret != True:
             break
       
        
       
        t1 = time.time()
        #image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        #boxes, confidence, classes = yolo.detect_image(image)
        frame = background
        indexes = []
        for i in range(len(boxes)):
            area = int(boxes[i][2]) * int(boxes[i][3])
            
            if area < 2500 or boxes[i][3] < 100	or boxes[i][2] < 50:
                indexes.append(i)
        for index in sorted(indexes, reverse=True):
            del boxes[index]
            del confidence[index]
            del classes[index]

        if tracking:
            
            #features = encoder(frame, boxes)
            features = encoder(np.zeros(frame.shape), boxes)
            
            detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                          zip(boxes, confidence, classes, features)]
            
        else:
            detections = [Detection_YOLO(bbox, confidence, cls) for bbox, confidence, cls in
                          zip(boxes, confidence, classes)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        if tracking:
            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                t_id = track.track_id
                print(bbox, 'tracker bbox')
                
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colors[t_id], 2)
                cv2.putText(frame, "ID:" + str(track.track_id) , (int(bbox[0]), int(bbox[1]) + 30 ), 0,
                            1.5e-3 * frame.shape[0], colors[t_id], 2)
                root_logger.info("ID:" + str(track.track_id) + " Position: " + str(bbox) + " Class: " + str(track.class_name))
                # + 30 to in(bbox[1]) 
        
        for det in detections:
            bbox = det.to_tlbr()
            #score = "%.2f" % round(det.confidence * 100, 2) + "%"
            #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            if len(classes) > 0:
                cls = det.cls
                cv2.putText(frame, str(cls) , (int(bbox[0]), int(bbox[3])), 0,
                            1.5e-3 * frame.shape[0], (0, 0, 255), 2)
                

        cv2.imshow('', frame)

        if writeVideo_flag: # and not asyncVideo_flag:
            
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            #if frame_index == 300:
            #    break
            

        fps_imutils.update()
        
        if not asyncVideo_flag:
            fps = (fps + (1./(time.time()-t1))) / 2
            print("FPS = %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        print('out written', out)
        out.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    #main(YOLO())
    main()
