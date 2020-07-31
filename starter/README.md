# Computer-Pointer-Controller
Uses 4 trained models and the users eye gaze direction to move the mouse pointer on the screen
***
### Computer Pointer Controller
The mouser.py app uses 4 trained network models to estimate the users gaze direction and then moves the cursor to 
that location on the screen. Although the user may use any video as input,the app is designed to use a cam facing the user. The 
app then extracts the users face from each frame and passes the face cutout as a new image to the next 2 models, one of which which detects 5 
facial landmarks from the users face (right eye, left eye, tip of the users nose and the right and left corners of the users mouth).
The other model estimates the (pose) yaw, pitch and roll (Euler Angles) of the users head. The data from these 2 models are passed to the final 
model which uses the images of the right and left eyes and the head pose (rotation angles) to estimate the direction of the users 
gaze. That information is then translated to a direction on the screen and the mouse is then movedin that direction. 
***
### models
Below is a list of the 4 models used for this project

        1. face-detection-adas-binary-0001 FP32-INT1
        
        2. head-pose-estimation-adas-0001 FP32
        
        3. landmarks-regression-retail-0009 FP32
        
        4. gaze-estimation-adas-0002 FP32
        


***
### Setup and Installation
Clone the git
if you do not have git...
sudo apt-get update && sudo apt-get install git

git clone 
***

Install Intels' OpenVINO 
### mouser.py requires OpenVINO 2020.1.023 and python3.5 or higher <------------------------------------------<
https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html
***

#### Linux
Refer to Install Intel Distribution of OpenVINO toolkit for Linux to learn how to install and configure the toolkit copy and paste the link below...
https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html

#### Windows
Refer to Install Intel Distribution of OpenVINO toolkit for Windows to learn how to install and configure the toolkit copy and paste the link below...
https://docs.openvinotoolkit.org/2020.1/_docs_install_guides_installing_openvino_windows.html
**Note: untested with windows

After cloning the repository and installing OpenVINO, cd to the 'models' folder. 
From within the folder, run...

        a.) (if you chose to install openvino in the default directory)
        /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name MODEL_NAME

        b.) (if you did not install openVINO in the default folder)
        **INSTALL_DIR**/deployment_tools/tools/model_downloader/downloader.py --name MODEL_NAME
        Substitute the names of the models used for this project for MODEL_NAME one at a time in the command above (see models section above for exeact model names).

#### Directory structure (after downloading models)


     starter/
        README.md
            test/
                gaze-estimationPERF_COUNTS_FP32.txt
                gaze-estimationPERF_COUNTS_FP16.txt
                gaze-estimationPERF_COUNTS.txt
                face-detection-binary_txt
                head-pose-estimation-PERF_FP16.txt
                head-pose-estimation-PERF_FP32.txt
                landmarks-regression-FP16.txt
                landmarks-regression-FP32.txt
                
            src/
                requirements.txt
                demo.mp4
                mouser.py
                mouse_controller.py
                input_feeder.py
            models/ **models will be empty after cloning, everything here and below will need to be added see above**
                intel/
                        face-detection-adas-binary-0001/
                                FP32-INT1/
                                        face-detection-adas-binary-0001.xml
                                        face-detection-adas-binary-0001.bin
                        gaze-estimation-adas-0002/
                                FP32/
                                        gaze-estimation-adas-0002.xml
                                        gaze-estimation-adas-0002.bin
                        head-pose-estimation-adas-0001/
                                FP32/
                                        gaze-estimation-adas-0002.xml
                                        gaze-estimation-adas-0002.bin
                        landmarks-regression-retail-0009/
                                FP32/
                                        landmarks-regression-retail-0009.xml
                                        landmarks-regression-retail-0009.bin"
                                        
                                        
                                        
***
### Create a virtual enviornment

        Install virtualenv
        pip3 install virtualenv==16.7.9

        from the src folder run...
        python3 -m venv pointer_venv
        
        To activate pointer venv from src folder run...
        source pointer_venv/bin/activate
        (it is suggested that mouser.py always be run from within the pointer_venv virtual enviornment)
         
### Install the dependencies
        from the src folder run...
        pip3 install -r requirements.txt

        
        Configure the environment to use the Intel Distribution of OpenVINO toolkit by exporting environment variables:
        If OpenVINO was installed in the default folder run...
        source /opt/intel/openvino/bin/setupvars.sh

#### RUN A DEMO
         To run the app with demo.mp4, from the src folder run...
         python3 src/mouser.py -t video -v True
         
         To run a the app using the cam, from the src folder run...
         python3 mouser.py
         
#### Run The application
        For the help message run...
        python3 mouser.py --help


        to run the application run...
        python3 mouser.py 

see --help for more info
***

### Documentation

'usage: mouser.py [-h] [-F FACE_MODEL] [-L LANDMARK_MODEL] [-H HEAD_POSE_MODEL]
                 [-G GAZE_MODEL] [-l CPU_EXTENSION] [-d DEVICE]
                 [-f INPUT_FILE] [-t INPUT_TYPE] [-v VISUAL] [-b BATCH]
                 [-i INFERENCE_TIMES]
                 

Options:
  -h, --help            Show this help message and exit.
  -F FACE_MODEL, --face_model FACE_MODEL
                        Optional. Path to the face-detection-adas-binary-0001
                        model. Only necessary if the default is not valid
                        
  -L LANDMARK_MODEL, --landmark_model LANDMARK_MODEL
                        Optional. Path to the landmarks-regression-retail-0009
                        model. Only necessary if the default is not valid
                        
  -H HEAD_POSE_MODEL, --head_pose_model HEAD_POSE_MODEL
                        Optional. Path to the head-pose-estimation-adas-0001
                        model. Only necessary if the default is not valid
                        
  -G GAZE_MODEL, --gaze_model GAZE_MODEL
                        Optional. Path to the gaze-estimation-adas-0002 model.
                        Only necessary if the default is not valid
                        
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional.Enter the path to a custom kernel shared
                        library is required for custom layers.
                        
  -d DEVICE, --device DEVICE
                        Optional. Default is CPU Specify the target device to
                        infer on; CPU or GPU.
                        
  -f INPUT_FILE, --input_file INPUT_FILE
                        if not cam enter the path to the input video file.
                        
  -t INPUT_TYPE, --input_type INPUT_TYPE
                        Optional. 'cam' is the default, Enter 'video' to
                        change the input to a file
                        
  -v VISUAL, --visual VISUAL
                        Optional. Default is false. Enter True to see visual
                        output of the individual models and disable mouse
                        movement. If false, the direction of the users gaze
                        will move the mouse cursor
                        
  -b BATCH, --batch BATCH
                        Optional. Default is one. Number of input frames per
                        inference. The app is only compatible with one frame
                        at a time
                        
  -i INFERENCE_TIMES, --inference_times INFERENCE_TIMES
                        Optional. Default is False. Will display inference
                        times on screen if True
'
                        

***

### Inference Time in seconds
|              Model                |  FP32    |    FP16   | FP32-INT8   | FP32-INT1 |
|-----------------------------------|----------|-----------|-------------|-----------|
|face-detection-adas-binary-0001    |  0.1652  | 0.1595    |    N/A      | 0.0708    |
|gaze-estimation-adas-0002          |  0.0085  | 0.0085    | *           |  N/A      |  
|landmarks-regression-retail-0009   |  0.0021  | 0.0016    | 0.0015      |  N/A      |
|head-pose-estimation-adas-0001     |  0.0079  | 0.0064    | *           |  N/A      |
*illegal instruction
||||

### Benchmarks
|              Model                       | Latency  |Throughput |  Load Time  | 
|------------------------------------------|----------|-----------|-------------|
|face-detection-adas-binary-0001 FP32-INT1 | 365.65ms | 11.51 FPS | 0.43s       |
|gaze-estimation-adas-0002  FP32           |  48.21ms | 79.78 FPS | 0.229s      | 
|gaze-estimation-adas-0002  FP16           |  38.59ms | 106.28 FPS| 0.261s      | 
|landmarks-regression-retail-0009 FP32     |  5.56ms  | 714.82 FPS| 0.1516s     |
|landmarks-regression-retail-0009 FP16     |  6.27ms  | 594.7 FPS | 0.1877s     |
|head-pose-estimation-adas-0001 FP32       | 40.54ms  | 94.42 FPS | 0.784s      |
|head-pose-estimation-adas-0001 FP16       |  35.87ms | 112.79 FPS| 0.221       |

From the data gathered above I found that the primary bottleneck was the face detection model. The FP32 and FP16 models latencies were disabling for the app. Interestingly the INT8 face detection model actually provided higher accuracy results that the FP16 or FP32. The landmark regression model was almost unnoticable as far as inference time or throughput is concerned. The ground truth accuracy of the landmark regression model seemed unchanged by model precision. Because the model had such a small impact on the overall performance of the app I decided to keep its precision at FP32 although, any of the model precisions would have been easily replaced. The head pose model and the gaze estimation model both provided similar results in that the difference between FP32 and FP16 was negligable in the current configuration. If I had more time I would look into streamlining the face detection model. A closer inspection of the execution of the model indicates that there were several ReLu and Conv2D layers that were not executed. Ideally I would have liked to push the FPS up to around 25 to 30 frames per second. See PERF_COUNTS files in the starter/test directory.

### Data Flow
                                    _________________________________________________
                                    |Video Input (file or cam) is read frame by frame|
                                    |_______Input Frame is fed to frst model_________|
                                                          |
                                                    face-detection
                                                          |
            |Model outputs an outcropping of the face detected in frame and passes a copy to 2 seperate models|
                                |                                                       |
                   |landmarks-regression|                                    |head-pose-estimation|
                ________________|___________________          __________________________|_______________________
               |model outputs 2 outcropped images of|        |Model outputs an estimation of the pose angles of | 
               |____of the left and right eyes______|        |the head in the image in Yaw Pitch and Roll format| 
                                             |________________________|
                                             |____gaze-estimation_____|
                                                         |
                                    Model outputs an estimation of the gaze as a vector
                                                
                           
                                
||||



### Edge Cases
I had a bit of difficulty trying to show the gaze direction accurately. While I tried many things I received the best results by taking the gaze vector out from the gaze estimation model and finding the dot product of it and its rotation matrix.Then I weighted the gaze with the head pose estimation. As a result of doing this some of the eye movements when used with demo.mp4 register but are not as accurate as I would of liked them to be. Although the visual of the gaze direction is stable and consistent with ground truth. 

Another issue I had with the output from the gaze estimation model is that it was inconsistent ranging from 0.1 to 10^30. Once the output is in a given range it remains in that range and after normalization it appears valid but it caused some issues with displaying the output and needed to accounted for. I resolved the issue by restarting main until the output is in an acceptable range. You may see this as the application startsup it will output an error message and restart until the output is corrected.

Lighting is crucial and not enough will cause complete failure. Although, the face detection model does, on occasion, output what it decides is a valid face. So where in the code the application breaks down is not necessarily the same from execution to execution. I accounted for this with a few try-except. If there is an issue with low light an error message is displayed and the application exits. 

The angle of incoming light along with the location of the camera can cause cursor movement that is unintended. In fact I found that without changing my gaze angle I can move the cursor by changing the angle of incoming light. I tried to make the application move the cursor to the point the users gaze meets the screen. However, with variations in the amount of light, the angle of incoming light and the location of the camera (relative to the screen) can change each individual users experience. Ideally, before this application is executed the introduction of a camera calibration matrix would help to pinpoint a more accurate representation of the location of the users gaze. I did create a rotation matrix but the increase to accuracy was marginal.
