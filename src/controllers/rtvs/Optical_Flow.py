"""
Calculates optical flow using the loaded model and 
saves the flow in corresponding folder
"""
import torch
import numpy as np
from PIL import Image
import cv2

from .flownet2.models import FlowNet2
from .utils.frame_utils import read_gen, flow_to_image


class Args:
    fp16 = False
    rgb_max = 255.0


class FlowNet2Utils:
    net = FlowNet2(Args()).cuda()
    net.load_state_dict(torch.load("../data/FlowNet2_checkpoint.pth.tar")["state_dict"])

    def __init__(self):
        pass

    @staticmethod
    def resize_img(img, new_size):
        return cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)


    @torch.no_grad()
    # passing the two images and then calculate the optical flow 

    def flow_calculate(self, img1, img2):
        """
        Reshapes the images and calculates the flow 
        """

        assert img1.shape == img2.shape, f"shapes dont match {img1.shape}, {img2.shape}"
        # reshape if not the same size 

        orig_shape = img1.shape[1::-1]
        req_shape = (512, 384)
        if orig_shape != req_shape:
            img1 = self.resize_img(img1, req_shape)
            img2 = self.resize_img(img2, req_shape)

        images = [img1, img2]
        # The operation converts from (samples, rows, columns, channels) into 
        #(samples, channels, rows, cols),maybe opencv to pytorch   
        images = np.array(images).transpose(3, 0, 1, 2)

        # converts to pytorch tensor 
        # adds a new dimension - BATCH SIZE : In this case , batch size is 1 !!
        im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

        # Flownet !!!
        result = self.net(im)[0].squeeze()

        # Rearrange image data from (channels, height, width) format to (height, width, channels) format
        data = result.data.cpu().numpy().transpose(1, 2, 0)

        if orig_shape != req_shape:# returns back in original shape 
            data = self.resize_img(data, orig_shape)# convert back to numpy 

        return data

    def flow_to_folder(self, name, flow): 
        """
        To just load the flow images into a folder 
        """
        f = open(name, "wb")
        f.write("PIEH".encode("utf-8"))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()

    def save_flow_with_image(self, folder):

        """
        Gets the source and image path and calculates the optical flow and loads into the 
        folder using the above class methods
        """
        img_source_path = folder + "/results/" + "test.rgb.00000.00000.png"
        img_goal_path = folder + "/des.png"
        img_src = read_gen(img_source_path)
        img_goal = read_gen(img_goal_path)

        f12 = self.flow_calculate(img_src, img_goal) # calculate the flow 

        self.flow_to_folder(folder + "/flow.flo", f12)

        flow_image = flow_to_image(f12)  #convert optical flow into color image

        im = Image.fromarray(flow_image)
        im.save(folder + "/flow.png")
