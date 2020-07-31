import cv2
from numpy import ndarray
import numpy as np
from numpy.random import default_rng
import pyautogui
from mouse_controller import MouseController

### setup the InputFeeder class
class InputFeeder:
    def __init__(self,input_type, input_file):
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
    
    def load_data(self):
        if self.input_type=='video':
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
        else:
            self.cap=cv2.VideoCapture(self.input_file)

    def next_batch(self, batch):
        while True:
            for _ in range(batch):
                _, frame=self.cap.read()
            yield frame
            
### Random number generator      
    def mkn(self,n):
        rng = default_rng()
        vals = rng.standard_normal(n)
        val = int(abs(vals*255))
        return val

### Close feed
    def close(self):
        if not self.input_type=='image':
            self.cap.release()
### Dot a marker           
    def ldot(self,frame,x,y):
        n = self.mkn(1)
        cv2.circle(frame,(x,y),4,(n*0.4,n,255),2)
        return frame

### Create an eye frame pic from frame    
    def eye_frame(self,frame,center):
        eye_box = 28
        eye_boxy = 22
        i = frame[(center['y']-eye_boxy):(center['y']+eye_boxy),(center['x']-eye_box):(center['x']+eye_box)]
        return i

### Create a list of landmarks    
    def face_points(self,box,shp,landm_out):
        landmarks = []
        right_eye = {'x':(int((box[0]+(landm_out[0]*shp[1])))),'y':(int((box[1])+(landm_out[1]*shp[0])))}
        landmarks.append(right_eye)
        left_eye = {'x':(int((box[0]+(landm_out[2]*shp[1])))),'y':(int((box[1])+(landm_out[3]*shp[0])))}
        landmarks.append(left_eye)
        nose = {'x':(int(box[0]+(landm_out[4]*shp[1]))),'y':(int(box[1]+(landm_out[5]*shp[0])))}
        landmarks.append(nose)
        right_mouth= {'x':(int(box[0]+(landm_out[6]*shp[1]))),'y':(int(box[1]+(landm_out[7]*shp[0])))}
        landmarks.append(right_mouth)
        left_mouth={'x':(int(box[0]+(landm_out[8]*shp[1]))),'y':(int(box[1]+(landm_out[9]*shp[0])))}
        landmarks.append(left_mouth)
        return landmarks

### Draw visual feedback from models            
    def visual(self,frame,frame_rx,frame_ry,frame_lx, frame_ly,x,y,resy,resp,h,w,box):
        offsetx = int((frame_rx+frame_lx)*0.5)
        offsety = int((frame_ry+frame_ly)*0.5)
        Hu = int(resy*offsety)
        Hv =int(offsety*resp)
        Hxx = int((Hu+(x*2))*0.3333)
        Hyy = int((Hv+(y*2))*0.3333)
        self.ldot(frame, frame_rx,frame_ry)
        self.ldot(frame, frame_lx,frame_ly)
        cv2.putText(frame, ('Gaze x: %.0f ' % Hxx), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 1,cv2.LINE_AA)
        cv2.putText(frame, ('Gaze y: %.0f ' % Hyy), (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 1,cv2.LINE_AA)
        cv2.putText(frame, ('Head Pose'), (offsetx+50, offsety-60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0,255), 1,cv2.LINE_AA)
        cv2.arrowedLine(frame,(frame_rx,frame_ry),(Hxx,Hyy), (0, 255,0), 3)
        cv2.rectangle(frame,(frame_rx-28,frame_ry-22),(frame_rx+28, frame_ry+22),(255,0,0))
        cv2.rectangle(frame,(frame_lx-28,frame_ly-22),(frame_lx+28, frame_ly+22),(255,0,0))
        cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(255,0,0))
        yaw = (np.deg2rad(resy))
        pitch = (np.deg2rad(resp))
        x_yaw = offsetx + (100*(yaw))
        y_yaw = offsety + (100*(pitch))
        frame = cv2.arrowedLine(frame, (offsetx,offsety),(x_yaw,y_yaw), (0,0,255), 5)
        
        
    def post_times(self,frame, gt,pt,lt,ft):
        cv2.putText(frame, ('Gaze inf time: %.6f ' % gt), (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 1,cv2.LINE_AA)
        cv2.putText(frame, ('Head pose inf time: %.6f ' % pt), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 1,cv2.LINE_AA)
        cv2.putText(frame, ('Landmarks inf time: %.6f ' % lt), (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 1,cv2.LINE_AA)
        cv2.putText(frame, ('Face inf time: %.6f ' % ft), (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 1,cv2.LINE_AA)
            
            
            

