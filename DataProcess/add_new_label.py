import os
from PIL import Image, ImageDraw, ImageFont
import random
import string

src_folder = './AllwithoutLabel' #替换为不带标记的文件夹路径
des_folder = './AllwithNewLabel2/' #替换为添加标记后的文件夹路径

# 如果目标文件夹不存在，则创建
if not os.path.exists(des_folder):
    os.makedirs(des_folder)
    

# 设置字体和大小
font_path = "Arial.ttf" 
font_size = 30
font = ImageFont.truetype(font_path, font_size)


# 设置标记颜色为宝蓝色
color = (14, 253, 254)

# # 第一种：设置标记文本'+'
text = "-"  # 可以替换为其他文本或使用随机生成的文本，如“--”

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
        
        #text = generate_random_letter() #采用第二种标注方式的话
        
        #获取文本的长宽
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        # 设置随机位置
        x = random.randint(0, image.width - text_width)
        y = random.randint(0, image.height - text_height)

        # 添加文本
        draw.text((x, y), text, fill=color, font=font)

    # 保存图像
    output_path = os.path.join(des_folder, filename)  # 替换为您想要保存的文件路径
    image.save(output_path)

print("所有图像处理完成。")