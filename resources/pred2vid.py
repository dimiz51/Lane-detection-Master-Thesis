import os
import cv2
import numpy as np
import torch

import sys
sys.path.insert(0,'../resources/')
from segnet_backbone import SegNet


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (448, 448)) # replace with the size of your input image
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def make_video(frames_dir, output_file):
    
    # get the list of frame filenames
    frame_filenames = [f for f in os.listdir(frames_dir) if f.endswith('.jpg') or f.endswith('.png')]
    # sort the filenames based on the integer value of the filename
    frame_filenames = sorted(frame_filenames, key=lambda x: int(x.split(".")[0]))

    # get the first frame to use as a reference for the video dimensions
    first_frame_path = os.path.join(frames_dir, frame_filenames[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, channels = first_frame.shape
    
    video_length = len(frame_filenames) * 0.1
    fps = len(frame_filenames) / video_length

    # initialize the video writer object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps , (width, height))
    
    # loop over the frames directory and process each frame
    for filename in frame_filenames:
        image_path = os.path.join(frames_dir, filename)
        image = cv2.imread(image_path)

        # write the image with the predicted lane points overlaid to the video
        video_writer.write(image)

    # release the video writer object
    video_writer.release()


if __name__ == '__main__':
    # cnn = SegNet()
    # cnn.load_weights('../models/best_segnet.pth')
    
    
    
    
    # # bad_lanes_dir = '../clips/bad_lanes_clip'
    # frames_dir = '../clips/good_lanes_clip'

    # # loop over the frames directory and process each frame
    # for filename in os.listdir(frames_dir):
    #     if filename.endswith(".jpg") or filename.endswith(".png"): 
    #         # load the image
    #         image_path = os.path.join(frames_dir, filename)
    #         image = torch.from_numpy(load_image(image_path).transpose(0,3, 1, 2))
    #         pred,_ = cnn.predict(image)
    #         mask = pred.squeeze(0).cpu().numpy()
       
    #         # convert the binary mask to green dots on the original frame
    #         original_image = cv2.imread(image_path)
    #         original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    #         green_dots = np.zeros_like(original_image)
    #         green_dots[:,:,1] = 255 * mask
    #         overlay_image = cv2.addWeighted(original_image, 0.7, green_dots, 0.3, 0)

    #         # save the image with the predicted lane points overlaid
    #         cv2.imwrite(f'../clips/pred_frames/{filename}', overlay_image)
    
    # Predicted marked frames dir
    pred_frames_dir = '../clips/pred_frames'
    output_file = '../clips/pred_vid_good.mp4'
    make_video(pred_frames_dir, output_file)