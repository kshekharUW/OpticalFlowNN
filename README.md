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


## **FAQ**

**Q:** What is Project Title?

**A:** Project Title is a project that does something useful.

**Q:** How do I install Project Title?

**A:** Follow the installation steps in the README file.

**Q:** How do I use Project Title?

**A:** Follow the usage steps in the README file.

**Q:** How do I contribute to Project Title?

**A:** Follow the contributing guidelines in the README file.

**Q:** What license is Project Title released under?

**A:** Project Title is released under the MIT License. See the **[LICENSE](https://www.blackbox.ai/share/LICENSE)** file for details.

## **Changelog**

- **0.1.0:** Initial release
- **0.1.1:** Fixed a bug in the build process
- **0.2.0:** Added a new feature
- **0.2.1:** Fixed a bug in the new feature

## **Contact**

If you have any questions or comments about Project Title, please contact **[Your Name](you@example.com)**.

## **Conclusion**

That's it! This is a basic template for a proper README file for a general project. You can customize it to fit your needs, but make sure to include all the necessary information. A good README file can help users understand and use your project, and it can also help attract contributors.
