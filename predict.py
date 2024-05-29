import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet
from torchvision.transforms import functional as F
import cv2


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 0  # exclude background
    weights_path = "./save_weights/model_0.pth"
    img_path = "./data/test/AC/16.png"
    # roi_mask_path = "./DRIVE/test/mask/01_test_mask.gif"
    # assert os.path.exists(weights_path), f"weights {weights_path} not found."
    # assert os.path.exists(img_path), f"image {img_path} not found."
    # assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    # mean = (0.709, 0.381, 0.224)
    # std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = UNet(in_channels=1, num_classes=classes+1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # load roi mask
    # roi_img = Image.open(roi_mask_path).convert('L')
    # roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert('L')

    # from pil image to tensor and normalize
    # data_transform = transforms.Compose([transforms.ToTensor(),
    #                                      transforms.Normalize(mean=mean, std=std)])
    # img = data_transform(original_img)
    img = F.to_tensor(original_img)

    # # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 1, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].squeeze(0)
        prediction = prediction.to("cpu").numpy()
        tensor = prediction.squeeze(0)  # 移除通道维度（C），因为灰度图只有一个通道
        array = tensor* 255  # 将归一化的值转换回[0, 255]范围
        array = array.astype('uint8')  # 转换为uint8类型
        pil_image = Image.fromarray(array)
        pil_image.show()
        pil_image.save("test_result.png")


if __name__ == '__main__':
    main()
