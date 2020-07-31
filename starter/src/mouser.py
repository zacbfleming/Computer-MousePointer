from openvino.inference_engine import IECore, IENetwork, IEPlugin
from time import time
import numpy as np
from input_feeder import InputFeeder
import logging as log
import cv2
import math
from mouse_controller import MouseController
import pyautogui
import argparse
from argparse import ArgumentParser, SUPPRESS
import sys



###initialize IECore and IENetworks
class Model_X:
    def __init__(self, model, device, batch):
        self.structure=model+'.xml'
        self.weights=model+'.bin'
        log.info("Loading network:\n\t{}\n\t{}".format(self.structure, self.weights))
        self.device=device
        self.batch=batch
        
### Return an executable model network
    def load_model(self):
        load = time()
        self.inet=IECore()
        self.enet=IENetwork(self.structure,self.weights)
        self.xnet=self.inet.load_network(self.enet,self.device,num_requests=0)
        print('Loadtime:', load-time())
        self.input_name=next(iter(self.enet.inputs))
        self.input_shape=self.enet.inputs[self.input_name].shape
        self.output_name=next(iter(self.enet.outputs))
        self.output_shape=self.enet.outputs[self.output_name].shape
        if "CPU" in args.device:
            yeslayers = self.inet.query_network(self.enet, "CPU")
            nolayers = [y for y in self.enet.layers.keys() if y not in yeslayers]
            log.info('Building model... \n\t{}'.format(self.structure))
            if len(nolayers) == 0:
                log.info("All layers are supported!")
                log.info("Device info:")
                versions = self.inet.get_versions(args.device)
                print("{}{}".format(" "*8, args.device))
                print("{}MKLDNNPlugin version ......... {}.{}".format(" "*8, versions[args.device].major, versions[args.device].minor))
                print("{}Build ........... {}".format(" "*8, versions[args.device].build_number))
            elif len(nolayers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".format(args.device, ', '.join(nolayers)))
                sys.exit(1)
        elif "GPU" in args.device:
            yeslayers = self.inet.query_network(self.enet, "GPU")
            nolayers = [y for y in self.enet.layers.keys() if y not in yeslayers]
            log.info('Building model... \n\t{}'.format(self.structure))
            if len(nolayers) == 0:
                log.info("All layers are supported!")
                log.info("Device info:")
                versions = self.inet.get_versions(args.device)
                print("{}{}".format(" "*8, args.device))
                print("{}libclDNNPlugin version ......... {}.{}".format(" "*8, versions[args.device].major, versions[args.device].minor))
                print("{}Build ........... {}".format(" "*8, versions[args.device].build_number))
            elif len(nolayers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".format(args.device, ', '.join(nolayers)))
                sys.exit(1)
        elif "MYRIAD" in args.device:
            yeslayers = self.inet.query_network(self.enet, "MYRIAD")
            nolayers = [y for y in self.enet.layers.keys() if y not in yeslayers]
            log.info('Building model... \n\t{}'.format(self.structure))
            if len(nolayers) == 0:
                log.info("All layers are supported!")
                log.info("Device info:")
                versions = self.inet.get_versions(args.device)
                print("{}{}".format(" "*8, args.device))
                print("{}libmyriadPlugin.so version ......... {}.{}".format(" "*8, versions[args.device].major, versions[args.device].minor))
                print("{}Build ........... {}".format(" "*8, versions[args.device].build_number))
            elif len(nolayers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".format(args.device, ', '.join(nolayers)))
                sys.exit(1)
        return self

### return the output of a generic model-Asynchronous
    def predict(self, image):
        infer_request_handle = self.xnet.start_async(request_id=0, inputs={self.input_name: image})
        infer_status = infer_request_handle.wait()
        result = infer_request_handle.outputs[self.output_name]       
        return result

### return the output from head-pose-estimation-ADAS-0001       
    def head_predict(self, image):
        infer_request_handle = self.xnet.start_async(request_id=0, inputs={self.input_name:image})
        infer_status = infer_request_handle.wait()
        resy = infer_request_handle.outputs['angle_y_fc']
        resp = infer_request_handle.outputs['angle_p_fc']
        resr = infer_request_handle.outputs['angle_r_fc']
        return resy, resp, resr       

### return the output from gaze-estimation-adas-0002     
    def gaze_predict(self,g):
        left_eye_image = g[2]
        right_eye_image = g[1]
        ypr=g[0]
        head_pose_angle = ypr.reshape(1,3)
        infer_request_handle = self.xnet.start_async(request_id=0, inputs={self.input_name:left_eye_image,self.input_name:right_eye_image,self.input_name:head_pose_angle})
        infer_status = infer_request_handle.wait()             
        res = infer_request_handle.outputs[self.output_name]
        return res

### return the processed image to model in batch-color-height-width format        
    def preprocess_input(self, image, hgt, wdt, b):
        try:
            p_frame = cv2.resize(image, (wdt,hgt))
            p_frame = p_frame.transpose((2,0,1))
            p_frame = p_frame.reshape(b,3,hgt,wdt)
            return p_frame
        except: pass  


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-F", "--face_model", help="Optional. Path to the face-detection-adas-binary-0001 model. Only necessary if the default is not valid", default='src/models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001')
    args.add_argument("-L", "--landmark_model", help="Optional. Path to the landmarks-regression-retail-0009 model. Only necessary if the default is not valid", default='src/models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009')
    args.add_argument("-H", "--head_pose_model", help="Optional. Path to the head-pose-estimation-adas-0001 model. Only necessary if the default is not valid", default='src/models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001')
    args.add_argument("-G", "--gaze_model", help="Optional. Path to the gaze-estimation-adas-0002 model. Only necessary if the default is not valid", default='src/models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002')
    args.add_argument('-l', "--cpu_extension",help="Optional.Enter the path to a custom kernel shared library is required for custom layers.",type=str, default=None)
    args.add_argument('-d', "--device",help="Optional. Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. A suitable plugin for device specified (CPU by default) will be used",default="CPU")
    args.add_argument('-f', '--input_file', help='if not cam enter the path to the input video file.', default='/home/artichoke/A_Udacity/inteldev/starter/src/demo.mp4')
    args.add_argument('-t', '--input_type', help='Optional. \'cam\' is the default, Enter \'video\' to change the input to a file', default='cam')
    args.add_argument('-v', '--visual', help='Optional. Default is false. Enter True to see visual output of the individual models and disable mouse movement. If false, the direction of the users gaze will move the mouse cursor', default=False,type=bool)
    args.add_argument('-b', '--batch', help='Optional. Default is one. Number of input frames per inference. The app is only compatible with one frame at a time', default=1)
    args.add_argument('-i', '--inference_times', help='Optional. Default is False. Will display inference times on screen if True', default=False)
    return parser





def rotation_mat(gaze_res):
    Rx = rotatez(gaze_res[0])
    Ry = rotatex(gaze_res[1])
    Rz = rotatey(gaze_res[2])
    Rm = (np.dot(Rx, (np.dot(Rz,Ry))))
    return Rm
    
def rotatex(xtheta):
    rox = np.asarray(([[1, 0, 0],[0, np.cos(xtheta), np.sin(-xtheta)],[0, np.sin(xtheta), np.cos(xtheta)]]), dtype=float)
    return rox

def rotatey(ytheta):
    roy = np.asarray(([[np.cos(ytheta), 0, np.sin(-ytheta)],[0,1,0], [np.sin(ytheta), 0, np.cos(ytheta)]]), dtype=float)
    return roy

def rotatez(ztheta):
    roz = np.asarray(([[np.cos(ztheta), np.sin(-ztheta), 0],[np.sin(ztheta), np.cos(ztheta),0], [0,0,1]]), dtype=float)
    return roz

def translation(gaze_vector,R):
    XYZ = np.dot(gaze_vector,R)
    cs = math.cos(XYZ[2]*(math.pi/180))
    sn = math.sin(XYZ[2]*(math.pi/180))
    XYZ = XYZ/np.linalg.norm(XYZ)
    tmpX = XYZ[0] * cs + XYZ[1] * sn
    tmpY = -XYZ[0] * sn + XYZ[1] * cs
    return (tmpX,tmpY),(XYZ)

        
def main(args):
    gaze_res2 = []
    vis=args.visual
    frame_count = 0
    start = time()
    xmin = 0
    ymin =0
    xmax =0
    ymax = 0
    vid = '/home/artichoke/A_Udacity/inteldev/starter/src/demo.mp4'
    batch = args.batch
    device = args.device
    input_type = args.input_type
    _F = Model_X(args.face_model,device,batch)
    fxnet = _F.load_model()
    _L = Model_X(args.landmark_model,device,batch)
    lxnet = _L.load_model()
    _H = Model_X(args.head_pose_model,device,batch)
    hxnet = _H.load_model()
    _G = Model_X(args.gaze_model,device,batch)
    gxnet = _G.load_model()

### Setup mouse controller class
    mickey = MouseController('low','fast')
    
### video management
    feed=InputFeeder(args.input_type, vid)
    feed.load_data()
    for frame in feed.next_batch(_G.batch):
        while feed.cap.isOpened():
            flag, frame = feed.cap.read()
            key_pressed = cv2.waitKey(60)
            if not flag:
                feed.close()
                exit()
            frame_count+=1
            init_w = feed.cap.get(3)
            init_h = feed.cap.get(4)
### performance counter
            t = time()

### Face detection
            f_frame = _F.preprocess_input(frame,384,672,_F.batch)
            try:
                face_ = time()
                face_out = _F.predict(f_frame)
            except:
                print('Error, Unable to detect a face, it is likely due to lighting. Try turning on a light or two.')
                feed.close()
                exit()
            face_inf = time() - face_
### define face in image/frame
            for det in face_out.reshape(-1, 7):
                conf = float(det[2])
                if conf >= 0.65:
                    box = []
                    xmin = int(det[3] * init_w)
                    ymin= int(det[4] * init_h)
                    xmax = int(det[5] * init_w)
                    ymax = int(det[6] * init_h)
                    box = [int(xmin),int(ymin),int(xmax),int(ymax)]

### Cutout face from image
            out_frame = frame[ymin:ymax,xmin:xmax]
            
### Process landmark model
            landm_frame = _L.preprocess_input(out_frame,48,48,_L.batch)
            land_ = time()
            landm_out = _L.predict(landm_frame)
            land_inf = time() - land_
            land_i = landm_out[0]
            
### Landmarks output
            shp = np.shape(out_frame)
            try:
                landmarks = feed.face_points(box,shp,land_i)
            except:
                print('Error, Unable to detect a face, it is likely due to lighting. Try turning on a light or two.')
                feed.close()
                exit()
            frame_rx = landmarks[0].get('x')
            frame_ry = landmarks[0].get('y')
            frame_lx = landmarks[1].get('x')
            frame_ly = landmarks[1].get('y')
    
### Process head pose model
            h_frame = _H.preprocess_input(out_frame,60,60,_H.batch)
            pose_ = time()
            resy,resp,resr = _H.head_predict(h_frame)
            pose_inf = time() -pose_

### setup inputs for gaze detection
            gaze_input = [ ]
            
### right eye and left eye frames
            right_eye_frame = feed.eye_frame(frame,landmarks[0])
            right_eye_image = _G.preprocess_input(frame,right_eye_frame,(60,60),_G.batch)
            left_eye_frame = feed.eye_frame(frame,landmarks[1])
            left_eye_image = _G.preprocess_input(frame,left_eye_frame,(60,60),_G.batch)
            
### head pose yaw pitch roll
            r_input = [ ]
            r_input=[resy,resp,resr]
            r_in = np.asarray(r_input, dtype=float)
            r_rad = np.deg2rad(r_in)

### Gaze input data
            gaze_input.append(r_in)
            gaze_input.append(right_eye_image)
            gaze_input.append(left_eye_image) 
            
### Gaze predict
            
            gaze_ = time()
            gaze_res = _G.gaze_predict(gaze_input)
           # if gaze_res2 == gaze_res[0][0]:
           #     cv2.destroyAllWindows()
           #     feed.close()
           #     main(args)
                
            gaze_inf = time()-gaze_ 
            print(gaze_res, gaze_res2)
            
            
            
### create rotation matrix translate and origin
            Rm = rotation_mat(r_rad)
            tmp, gaze_vector = translation(gaze_res[0],Rm)
            offsety = int((frame_ry+frame_ly)*0.5)
            offsetx = int((frame_rx+frame_lx)*0.5)

### move cursor to origin
            if frame_count is 1:
                pyautogui.moveTo(offsetx,offsety)
                gaze_res2 = gaze_res[0][0]
            

### Set visual X Y variables
            X = int((gaze_vector[0]*init_w)+offsetx)
            Y = int((gaze_vector[1]*init_h)+offsety)
            if args.inference_times is True:
                feed.post_times(frame, gaze_inf, pose_inf, land_inf, face_inf)
            
### Move mouse or draw a display plus mickey
            if vis is True:
                feed.visual(frame,frame_rx,frame_ry,frame_lx, frame_ly,X,Y,resy,resp,init_h, init_w, box)
            if args.input_type =='cam':
                mickey.move(-gaze_vector[0],gaze_vector[1])
            elif args.input_type =='video':
                mickey.move(gaze_vector[0],gaze_vector[1])

            cv2.imshow('input',frame)
            if key_pressed == 27:
                cv2.waitKey(60)
           
                cv2.destroyAllWindows()
                feed.close()
                exit(0)
           
            
### Call main setup args           
if __name__=='__main__':
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    main(args)
