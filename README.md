# **README File**

## **Optical Flow**

Optical Flow guided near field brightning

## **Introduction**

This project aims to enhance video motion by estimating optical flow to identify regions of significant movement. These regions will be brightened, with the intensity of the brightening modulated by depth information. This will create a visually striking effect, where motion closer to the camera appears more pronounced than distant motion.

## **Environment Setup/Requirements**
1. Ubuntu 22.04
2. Docker
3. Visual Studio Code (for convenience)
4. Install Dev Containers extension (to attach to a running container). Good youtube guide: https://www.youtube.com/watch?v=dihfA7Ol6Mw

## **Usage**
Follow these steps:
1. Clone the repository: **`https://github.com/kshekharUW/OpticalFlowNN.git`**
2. Navigate to the project directory: **`OpticalFlowNN`**
3. Run the script: **`./run.sh`**. This should launch a container named raftFlowContainer and open up VS code. If you don't want to use VS code, you can use "docker exec" to run commands inside the container.
5. Place input video inside volume/RAFT/videos

## **License**

Project Title is released under the MIT License.

## **Authors and Acknowledgment**

Project Title was created by **[Krishnendu Shekhar](https://github.com/kshekharUW)**.

Project draws heavily from the following repos:
- **https://github.com/princeton-vl/RAFT.git**
- **https://github.com/spmallick/learnopencv.git**


## **Approach**
Overall, this code iterates through a video, preprocesses frames, predicts optical flow using a pre-trained model, and optionally visualizes the flow on the frame. It skips frames based on a user-defined factor and measures the execution time for the prediction.
  1. Preprocessing: The Preprocessing step takes an image frame and resizes it while maintaining aspect ratio to a size compatible with the deep learning model. It then converts the image to a PyTorch tensor and transfers it to the specified device. The function also returns the scaling factor used during resizing.
  2. Optical Flow prediction: This step is performed via the OpticalFlowProcessor class. The OpticalFlowProcessor class is designed to analyze videos and visualize motion. It loads a pre-trained RAFT model for optical flow estimation, processes video frames, predicts optical flow, and visualizes the results. The visualization function can optionally incorporate depth information to highlight motion in specific regions. The class provides a flexible framework for video analysis and visualization tasks.
  3. Brightning: This step is performed via the brighten_image function. The brighten_image function enhances the brightness of an RGB image based on a given brightness map. This map indicates the desired level of brightening for each pixel. The function offers several methods for brightening: additive, multiplicative, gamma correction, and blending. The additive method simply adds a scaled version of the brightness map to the image. The multiplicative method scales the image pixels by a factor based on the brightness map. Gamma correction adjusts the overall image contrast. The blending method combines the original image and a brightened version weighted by the brightness map. The function ensures that the output image pixel values remain within the valid range of 0 to 255.    



## **Changelog**

- **0.1.0:** Initial release
- **0.1.1:** Fixed a bug in the build process
- **0.2.0:** Added a new feature
- **0.2.1:** Fixed a bug in the new feature

## **Contact**

If you have any questions or comments about Project Title, please contact **[Your Name](you@example.com)**.

## **Conclusion**

That's it! This is a basic template for a proper README file for a general project. You can customize it to fit your needs, but make sure to include all the necessary information. A good README file can help users understand and use your project, and it can also help attract contributors.
