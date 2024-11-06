# **README**

## **Optical Flow Brightening**

Optical Flow guided near field brightning

## **Introduction**

This project aims to enhance video motion by estimating optical flow to identify regions of significant movement. These regions will be brightened, with the intensity of the brightening modulated by depth information. This will create a visually striking effect, where motion closer to the camera appears more pronounced than distant motion.

## **Environment Setup/Requirements**
1. Ubuntu 22.04
2. Docker
3. Visual Studio Code (for convenience)
4. Install Dev Containers extension (to attach to a running container). Good [YouTube guide](https://www.youtube.com/watch?v=dihfA7Ol6Mw).

## **Usage**
Follow these steps:
1. Clone the repository: **`https://github.com/kshekharUW/OpticalFlowNN.git`**
2. Navigate to the project directory: **`OpticalFlowNN`**
3. Run the script: **`./run.sh`**. This should launch a container named raftFlowContainer and open up VS code. If you don't want to use VS code, you can use "docker exec" to run commands inside the container.
5. Place input video inside volume/RAFT/videos
6. Download and extract source_depth inside videos folder.

## **Demo**
[Output](https://drive.google.com/file/d/12GZgBbRX-SGOZLLjVHnbjzGZ2KnO9Ufi/view?usp=sharing)
- There are four images in each frame. [input frame, output frame, optical flow, brightning mask]
-![frame_317](https://github.com/user-attachments/assets/6afbffb2-7776-4057-90b5-abfce91f4b00)
-![frame_1](https://github.com/user-attachments/assets/058d41a7-76fb-4d06-8b07-f96a685fa717)


## **Approach**
Overall, this code iterates through a video, preprocesses frames, predicts optical flow using a pre-trained model, and optionally visualizes the flow on the frame. It skips frames based on a user-defined factor and measures the execution time for the prediction.
  1. Preprocessing: The Preprocessing step takes an image frame and resizes it while maintaining aspect ratio to a size compatible with the deep learning model. It then converts the image to a PyTorch tensor and transfers it to the specified device. The function also returns the scaling factor used during resizing.
  2. Optical Flow prediction: This step is performed via the OpticalFlowProcessor class. The OpticalFlowProcessor class is designed to analyze videos and visualize motion. It loads a pre-trained RAFT model for optical flow estimation, processes video frames, predicts optical flow, and visualizes the results. The visualization function can optionally incorporate depth information to highlight motion in specific regions. The class provides a flexible framework for video analysis and visualization tasks.
  3. Brightning: This step is performed via the brighten_image function. The brighten_image function enhances the brightness of an RGB image based on a given brightness map. This map indicates the desired level of brightening for each pixel. The function offers several methods for brightening: additive, multiplicative, gamma correction, and blending. The additive method simply adds a scaled version of the brightness map to the image. The multiplicative method scales the image pixels by a factor based on the brightness map. Gamma correction adjusts the overall image contrast. The blending method combines the original image and a brightened version weighted by the brightness map. The function ensures that the output image pixel values remain within the valid range of 0 to 255. 

## **Metrics**
On a NVIDIA GTX 1070 GPU, the processing time per frame was ~0.05 s. 

## **Limitations**
Given the movement of the camera, parts of the ground and sidewalk have large apparent motion relative to the camera; resulting in high optical flow results from the model.
-![frame_16](https://github.com/user-attachments/assets/f3a3250f-2a77-442c-b50b-68745d31c98b)


## **Possible Improvements**
1. Pre-inference on the RAFT model, we can perform image alignment/registration to undo the motion of the camera.
2. We can perform background subtraction pre-inference and perform optical flow on foreground objects only.
3. If we have access to camera intrinsic matrix, given we have a depth map available for each frame, we can use the object coordinates to estimate the pose of the camera for each frame. We can undo pose changes bewteen consequtive frames and improve our optical flow results.

## **Authors and Acknowledgment**
Optical Flow Brightning was created by **[Krishnendu Shekhar](https://github.com/kshekharUW)**.

Project draws heavily from the following repos:
- **https://github.com/princeton-vl/RAFT.git**
- **https://github.com/spmallick/learnopencv.git**
