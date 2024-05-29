import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import string
import cv2

src_folder = './AllwithoutLabelRGB' #替换为不带标记的文件夹路径
des_folder = './AllwithLabel3/' #替换为添加标记后的文件夹路径

# 如果目标文件夹不存在，则创建
if not os.path.exists(des_folder):
    os.makedirs(des_folder)
    

# 设置字体和大小
font_path = "Arial.ttf" 
font_size = 20
font = ImageFont.truetype(font_path, font_size)


# 设置标记颜色为宝蓝色
color = (14, 253, 254)

# 设置旋转角度
angle_range = (-180, 180)  # 设置旋转角度范围

# # 第一种：设置标记文本'+'
text = "+"  # 可以替换为其他文本或使用随机生成的文本，如“--”

# 第二种：设置标记文本为随机字母
# 生成随机字母的函数
def generate_random_letter():
    return random.choice(string.ascii_letters)  # ascii_letters包含大小写字母


for filename in os.listdir(src_folder):
    # 构建完整的文件路径
    file_path = os.path.join(src_folder, filename)
    
    # 打开图像文件
    image = Image.open(file_path)
    draw = ImageDraw.Draw(image)
    
    # 随机添加100次标记
    for i in range(100):
        
        # text = generate_random_letter() #采用第二种标注方式的话
        
        # 获取文本的掩码
        mask = Image.new('1', font.getsize(text), 0) #标记0设置掩码背景无颜色
        draw = ImageDraw.Draw(mask)
        draw.text((0, 0), text, fill=1, font=font) #标记1设置填充文本有颜色
        
        # 将掩码旋转指定角度
        angle = random.randint(*angle_range)
        rotated_mask = mask.rotate(angle, expand=1)
        
        # 找到旋转后文本的左上角位置
        w, h = rotated_mask.size
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = random.randint(0, image.width - w)
        y = random.randint(0, image.height - h)
        
        # 将旋转的文本添加到图像上
        image.paste(color, mask=rotated_mask, box=(x, y))
       
    # 转换为灰度图像
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # 保存图像
    output_path = os.path.join(des_folder, filename)  # 替换为您想要保存的文件路径
    # image.save(output_path)
    cv2.imwrite(output_path, gray_image)

print("所有图像处理完成。")