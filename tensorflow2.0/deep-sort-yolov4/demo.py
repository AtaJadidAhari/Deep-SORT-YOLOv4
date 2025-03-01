#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import sys

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from PIL import Image
from yolo import YOLO
import random

import zmq
import pickle
import zlib

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
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


import logging# Starting a logger for the server

root_logger = logging.getLogger('Logger')
root_logger.setLevel(logging.DEBUG)

handler = logging.FileHandler('logger.txt')
handler.setLevel(logging.DEBUG)
log_format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
handler.setFormatter(log_format)
root_logger.addHandler(handler)





def main(yolo):
    frame_index = 0
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    two_cameras = False
    same_view = False

    recieving = False
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]
    
    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    tracking = True
    writeVideo_flag = True
    asyncVideo_flag = False

    file_path = '/home/ata/Desktop/mobile.mp4'
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
            print('here', w, h)
            #fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            h = 480
            w = 640
            out = cv2.VideoWriter('output_yolov4.avi', fourcc, 30, (w, h))
            frame_index = -1
    

    
    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
    
    if two_cameras is True:

        cap2 = cv2.VideoCapture("/home/ata/Desktop/second dataset/cam1_4.avi")
    flag = True
    while True:
        
        if not recieving:
            ret, frame = video_capture.read()  # frame shape 640*480*3
        else:
            
            #rpi_name, frame = image_hub.recv_image()
            #image_hub.send_reply(b'OK')
            #background = Image.new('RGBA', (w, h), (255, 255, 255, 255))
            background = np.zeros((480, 640, 3)).astype(np.uint8)
          
            #background = Image.new('RGB', (480, 640))
            z = socket.recv(flags=0)
            p = zlib.decompress(z)
            detections = pickle.loads(p)
            i = 0
            for p in detections:
                bbox = [int(i) for i in p[1]]
                
                #p[0] = Image.fromarray(p[0][...,::-1]) 
                
                print(p[0].shape, i)
                i += 1
                print(bbox)
                background[ int(bbox[1]):int(bbox[3]), min(h, int(bbox[0])):int(bbox[2]), :] = p[0]
                
                #background = Image.fromarray(background[...,::-1].astype(np.uint8))
                #background.paste(p[0], (10, 20, 30, 40))
            cv2.imshow('opx', background)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        #if frame_index < 700:
        #    frame_index += 1
        #    continue
        ret = True
        if ret != True:
             break

        if two_cameras is True:
            
            ret2, frame2 = cap2.read() 
            print(frame2)
            if ret2 != True:
                break
            #frame2 = cv2.resize(frame2, (640, 480), interpolation=cv2.INTER_AREA)
          
        else:
            #frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            ret2 = True
            
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        if two_cameras is True and same_view is True:
            frame = np.concatenate((frame, frame2), axis=1)   
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            out = cv2.VideoWriter('output_yolov4.avi', fourcc, 30, (frame.shape[1], frame.shape[0]))
            frame_index = -1
        
        if flag is True and two_cameras is True:
            print(frame,)
            h, p_tuple = matching.get_homography(frame, frame2, 0.75, 1000)
            #p_tuple = (404, -591, 2041, 721)
            #h = np.array([[ 1.33277738e-01,  1.41078549e-01,  1.82398971e+02], [-9.41189709e-02,  5.82029384e-01, -1.73458913e+01], [-5.01171701e-04,  3.08630046e-04,  4.51327067e-01]])
            min_x = min(0, p_tuple[0])
            min_x = p_tuple[0]

            max_x = max(frame.shape[1], p_tuple[2])
            min_y = min(0, p_tuple[1])
            min_y = p_tuple[1]
            
            max_y = max(frame.shape[0], p_tuple[3])
            f_shape = (max_x - min_x + 1, max_y - min_y + 1)
            offset_mat = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
            h = np.matmul(offset_mat, h)
            f_shape = (max_x - min_x, max_y - min_y)
            tr_frame2 = cv2.warpPerspective(frame2, h, f_shape)
            tr_frame1 = np.zeros(tr_frame2.shape)
            tr_frame1[-min_y: frame.shape[0] - min_y, -min_x: frame.shape[1] - min_x, :] = frame
            mask = panorama.merger(tr_frame1, tr_frame2)
            

            frame3 = cv2.warpPerspective(frame2, h, f_shape).astype(np.uint8)
            frame4 = np.zeros(frame3.shape)
            frame4[-min_y: frame.shape[0] - min_y, -min_x: frame.shape[1] - min_x, :] = frame
            frame3 = mask*frame4 + (1-mask)*frame3
            frame3 = frame3.astype(np.uint8)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            out = cv2.VideoWriter('output_yolov4.avi', fourcc, 30, (frame3.shape[1], frame3.shape[0]))
            frame_index = -1
            flag = False
            
        if two_cameras is True:
            frame3 = cv2.warpPerspective(frame2, h, f_shape).astype(np.uint8)
            frame4 = np.zeros(frame3.shape)
            frame4[-min_y: frame.shape[0] - min_y, -min_x: frame.shape[1] - min_x, :] = frame
            frame3 = mask*frame4 + (1-mask)*frame3
            frame3 = frame3.astype(np.uint8)
            #ret, frame = cap.read()
            #ret, frame_2 = cap2.read()
            #frame = np.concatenate((frame, frame_2), axis=1)
            frame = frame3
        

        t1 = time.time()

        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxes, confidence, classes = yolo.detect_image(image)

        
        


        background = np.zeros((h, w, 3)).astype(np.uint8)
        #print(boxes, frame_index)
        #for bbox in boxes:
                
                
                #p[0] = Image.fromarray(p[0][...,::-1]) 
                
               
                
        #        background[int(bbox[1]):int(bbox[1]) +int(bbox[3]) ,int(bbox[0]) : int(bbox[0]) + int(bbox[2]), :] = frame[int(bbox[1]):int(bbox[1]) +int(bbox[3]) ,int(bbox[0]) : int(bbox[0]) + int(bbox[2]), :]
        #frame = background
        indexes = []
        
        for i in range(len(boxes)):
            area = int(boxes[i][2]) * int(boxes[i][3])
            if area < 4000 or boxes[i][3] < 100	or boxes[i][2] < 50:
                indexes.append(i)
        for index in sorted(indexes, reverse=True):
            del boxes[index]
            del confidence[index]
            del classes[index]
 
        for b in boxes: 
            print(b[3], b[2])

            print(int(b[2]) * int(b[3]), 'area')
        
        
        if tracking:
            features = encoder(frame, boxes)
            print(features, 'features')
            #features = encoder(np.zeros(frame.shape), boxes)
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
        
        log = []
        if tracking:
            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                
                t_id = track.track_id
                
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colors[t_id], 2)
                cv2.putText(frame, "ID:" + str(track.track_id), (int(bbox[0]), int(bbox[1]) + 30 ), 0,
                            1.5e-3 * frame.shape[0], colors[t_id], 2)
                root_logger.info("ID:" + str(track.track_id) + " Position: " + str(bbox) + " Class: " + str(track.class_name))
                
                # + 30 to in(bbox[1]) 
        i = 0
        
        for det in detections:
            bbox = det.to_tlbr()
            #score = "%.2f" % round(det.confidence * 100, 2) + "%"
            #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            if len(classes) > 0:
                
                cls = det.cls
                cv2.putText(frame, str(cls) , (int(bbox[0]), int(bbox[3])), 0,
                            1.5e-3 * frame.shape[0], (0, 255, 0), 1)
                #if i < len(log):
                
                #    root_logger.info(log[i] + "Class Number: " , str(cls))
            
            i += 1
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
            print("FPS = %f"%(fps), frame_index)
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    #if writeVideo_flag:
    #    print('out written', out)
    #    out.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
    
