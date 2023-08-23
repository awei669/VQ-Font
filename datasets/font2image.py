from ttf_utils import *
from tqdm import tqdm

char2img_list = '才下寸大丈与万上小口山巾千乞川亿个夕久么勺凡丸及广亡门丫义之尸己已巳弓子卫也女刃飞习叉马乡'
# Characters to be generated, including characters for training and testing.

font_file = 'path/to/save/ttf/'
image_file = 'path/to/save/images/'
fonts = os.listdir(font_file)

for font in tqdm(fonts):
    font_path = os.path.join(font_file, font)
    print(font_path)
    try:
        font2image(font_path, image_file, char2img_list, 128)
        print(font)
    except Exception as e:
        print(e)
remove_empty_floder(image_file)
