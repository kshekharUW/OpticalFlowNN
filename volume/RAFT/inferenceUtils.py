import os
import cv2
import numpy as np
import torch
import math
import time
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from collections import OrderedDict

from raft import RAFT
from utils import flow_viz


def generate_frame_name(count):
    """Generates a frame name in the format "frame_<7-digit-number>.png".

    Args:
        count: The frame number.

    Returns:
        The generated frame name.
    """

    frame_number = str(count).zfill(7)  # Zfill pads with zeros to 7 digits
    return f"frame_{frame_number}.png"


def frame_preprocess(frame, device, max_width=float("inf")):
    """
    Preprocesses a frame by resizing and converting it to a PyTorch tensor.

    Args:
        frame: NumPy array representing the image frame.
        device: Device to transfer the tensor to (e.g., "cuda" for GPU).

    Returns:
        A PyTorch tensor representing the preprocessed frame.
    """

    # Get original image dimensions
    original_height, original_width = frame.shape[:2]

    # Resize frame while maintaining aspect ratio and ensuring dimensions are multiples of 8
    new_width = min(max_width, original_width)
    new_width = math.ceil(new_width / 8) * 8
    scale = new_width / original_width
    new_height = (math.ceil(original_height * scale) // 8) * 8

    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Convert to PyTorch tensor
    resized_frame = resized_frame.transpose(2, 0, 1)
    resized_frame = torch.from_numpy(resized_frame).float()
    resized_frame = resized_frame.unsqueeze(0)
    resized_frame = resized_frame.to(device)

    return resized_frame, scale


def getDepthMap(count, flow, depthFolder="./videos/source_depth/", visualize=False):
    """Loads a depth map from the specified folder, normalizes, inverts, and optionally visualizes it alongside flow magnitude, a weighted mask, and depth difference.

    Args:
    count: The frame number.
    flow: A 2-channel image representing the flow.
    depthFolder: The path to the depth map folder.
    visualize: Whether to visualize the depth map and flow channels.

    Returns:
    The loaded, normalized, and inverted depth map as a NumPy array.
    """

    frame_name = generate_frame_name(count)
    depth_path = os.path.join(depthFolder, frame_name)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # Normalize depth maps to 0-1 range and invert
    depth_map = depth_map.astype(np.float32) / np.max(depth_map)
    depth_map = 1 - depth_map
    depth_map[depth_map < np.max(depth_map) * 0.75] = 0

    # Resize depth maps for consistent display
    depth_map = cv2.resize(
        depth_map, (flow.shape[1], flow.shape[0]), interpolation=cv2.INTER_LINEAR
    )

    # # Calculate top 30% flow_magnitude
    flow_magnitude = np.sqrt(np.square(flow[:, :, 0]) + np.square(flow[:, :, 1]))
    orig_flow_magnitude = flow_magnitude.copy()
    threshold = np.percentile(flow_magnitude.flatten(), 70)
    flow_magnitude[flow_magnitude < threshold] = 0
    flow_magnitude = flow_magnitude.astype(np.float32) / np.max(flow_magnitude)
    # flow_magnitude[flow_magnitude< np.max(flow_magnitude*0.5)] = 0

    flow_magnitude[depth_map < 0.1] = 0

    if visualize:
        # Create a figure for subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 12))

        # Plot depth map
        im1 = axes[0].imshow(depth_map, cmap="inferno")
        axes[0].set_title("Depth Map")
        axes[0].axis("off")
        fig.colorbar(im1, ax=axes[0], orientation="horizontal")

        # Plot base flow magnitude
        im2 = axes[1].imshow(orig_flow_magnitude, cmap="inferno")
        axes[1].set_title("Flow Magnitude Orig")
        axes[1].axis("off")
        fig.colorbar(im2, ax=axes[1], orientation="horizontal")

        # Plot flow magnitude
        im3 = axes[2].imshow(flow_magnitude, cmap="inferno")
        axes[2].set_title("Flow Magnitude")
        axes[2].axis("off")
        fig.colorbar(im3, ax=axes[2], orientation="horizontal")

        plt.suptitle(f"Visualization for Frame {count}")
        plt.tight_layout()
        plt.show()

    return flow_magnitude * 255


def resize_flow_to_match_image(flow, img):
    """Resizes the flow to match the dimensions of the image, handling different channel numbers.

    Args:
      flow: A NumPy array representing the flow image.
      img: A NumPy array representing the original image.

    Returns:
      The resized flow image.
    """

    img_height, img_width, img_channels = img.shape
    flow_height, flow_width, flow_channels = flow.shape

    if flow_height != img_height or flow_width != img_width:
        # Resize the flow image, preserving the number of channels
        resized_flow = cv2.resize(
            flow, (img_width, img_height), interpolation=cv2.INTER_LINEAR
        )
    else:
        return flow

    return resized_flow


def brighten_image(img, brighten_area, method="multiplicative"):
    """Brightens an RGB image based on a brightness map.

    Args:
      img: The input RGB image.
      brighten_area: The brightness map, a single-channel image with values between 0 and 255.
      method: The brightening method to use, one of 'additive', 'multiplicative', 'gamma', or 'blend'.

    Returns:
      The brightened RGB image.
    """

    # Ensure the images have the same shape
    assert img.shape[:2] == brighten_area.shape

    # Normalize brightness map to 0-1 range
    brighten_area = brighten_area / 255.0

    if method == "additive":
        brightened_img = np.clip(img + brighten_area * 255, 0, 255).astype(np.uint8)
    elif method == "multiplicative":
        brighten_area = np.dstack([brighten_area] * 3)
        brightened_img = np.clip(img * (1 + brighten_area), 0, 255).astype(np.uint8)
    elif method == "gamma":
        gamma = 4.0  # Adjust gamma value as needed
        brightened_img = np.clip((img / 255.0) ** gamma * 255, 0, 255).astype(np.uint8)
    elif method == "blend":
        # Convert brightness map to a 3-channel image
        brighten_area = np.dstack([brighten_area] * 3)
        brightened_img = np.clip(
            img * (1 - brighten_area) + img * brighten_area, 0, 255
        ).astype(np.uint8)
    else:
        raise ValueError(
            "Invalid method. Please choose from 'additive', 'multiplicative', 'gamma', or 'blend'."
        )

    return brightened_img


class OpticalFlowProcessor:
    def __init__(self, args, max_width=1024):
        self.args = args
        self.max_width = max_width

        # Load model and weights
        self.model = None
        self.device = None
        self._load_model()

        # Create output directory if saving frames
        if self.args.save:
            if not os.path.exists("demo_frames"):
                os.mkdir("demo_frames")

    def _load_model(self):
        # get the RAFT model
        self.model = RAFT(self.args)
        # load pretrained weights
        pretrained_weights = torch.load(self.args.model)

        if torch.cuda.is_available():
            print("CUDA is available!")
            self.device = "cuda"
            self.model = torch.nn.DataParallel(self.model)
            self.model.load_state_dict(pretrained_weights)
            self.model.to(self.device)
        else:
            print("No CUDA!")
            self.device = "cpu"
            # change key names for CPU runtime
            pretrained_weights = self.get_cpu_model(pretrained_weights)
            self.model.load_state_dict(pretrained_weights)

        # change model's mode to evaluation
        self.model.eval()

    def _get_cpu_model(model):
        new_model = OrderedDict()
        # get all layer's names from model
        for name in model:
            # create new name and update new model
            new_name = name[7:]
            new_model[new_name] = model[name]
        return new_model

    def vizualize_flow(self, img, flo, counter):
        # permute the channels and change device is necessary
        # print(f"Pre Permute Img: {img.shape} Flow: {flo.shape}")

        img = img[0].permute(1, 2, 0).cpu().numpy()
        flo = flo[0].permute(1, 2, 0).cpu().numpy()
        # print(f"Post permute: {img.shape} Flow: {flo.shape}")

        flo = resize_flow_to_match_image(flo, img)
        # print(f"Post resize: {img.shape} Flow: {flo.shape}")

        # get top 10% mask
        top_mask = getDepthMap(counter - 1, flo, visualize=False)
        # Create a 3-channel image by stacking the 1-channel image 3 times for viewing
        top_mask_3channel = np.stack([top_mask] * 3, axis=-1)

        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        flo = cv2.cvtColor(flo, cv2.COLOR_RGB2BGR)
        # print(f"RGB Img: {img.shape} Flow: {flo.shape}")

        bright_image = brighten_image(img, top_mask)

        # concatenate, save and show images
        img_flo = np.concatenate([img, bright_image, flo, top_mask_3channel], axis=1)
        # img_flo = np.concatenate([img, bright_image], axis=1)

        # Calculate desired width to maintain aspect ratio
        img_height, img_width, _ = img_flo.shape
        desired_height = 1080
        desired_width = int(img_width * desired_height / img_height)

        # Resize the combined image to the desired window size
        img_flo = cv2.resize(img_flo, (desired_width, desired_height))

        if self.args.save:
            cv2.imwrite(f"demo_frames/frame_{str(counter)}.jpg", img_flo)
        cv2.imshow("Optical Flow", img_flo / 255.0)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            return False
        return True

    def run_inference(
        self,
        video_path=None,
        start_frame=0,
        end_frame=float("inf"),
        skip_factor=10,
    ):
        if video_path == None:
            video_path = self.args.video

        # capture the video and get the first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame_1 = cap.read()
        print(f"Read image of type {type(frame_1)} and shape {frame_1.shape}")

        counter = 0
        max_width = 1024  # Maximum allowed width for resized images

        with torch.no_grad():
            while True:
                ret, frame_2 = cap.read()
                if not ret:
                    break
                counter += 1

                # Skip frames based on skip_factor (except the first frame)
                if counter > 1 and counter % skip_factor != 0:
                    continue
                print(f"Counter : {counter} ", end="")

                start_time = time.time()
                # Preprocess the resized frames
                resized_frame_1, scale = frame_preprocess(
                    frame_1, self.device, max_width=max_width
                )
                resized_frame_2, scale = frame_preprocess(
                    frame_2, self.device, max_width=max_width
                )

                # Predict the flow on resized images
                print(
                    f"Input: {frame_1.shape} Intermediate: {resized_frame_1.shape} ",
                    end="",
                )
                flow_low, flow_up = self.model(
                    resized_frame_1,
                    resized_frame_2,
                    iters=self.args.iters,
                    test_mode=True,
                )
                frame_1 = torch.from_numpy(frame_1)
                frame_1 = frame_1.permute(2, 0, 1).unsqueeze(0)

                end_time = time.time()
                execution_time = end_time - start_time
                print(
                    f"Output: {flow_up.shape}, scale: {scale} Exec time: {execution_time:.2f} s"
                )

                # Transpose the flow output and convert it into numpy array
                ret = self.vizualize_flow(frame_1, flow_up, counter)
                if not ret:
                    break

                frame_1 = frame_2
