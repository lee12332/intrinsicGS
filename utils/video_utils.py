import cv2
from tqdm import trange
import os
import numpy as np

import os
import cv2

FRAME_RATE = 30

def convert_file_to_video(file_dir, video_dir, no_alpha=False):
    if not os.path.exists(file_dir):
        raise ValueError(f"Input directory '{file_dir}' does not exist.")
    
    image_names = sorted(os.listdir(file_dir))
    print('Input Directory:', file_dir)
    print('Output Video Path:', video_dir)

    format_tile = '.mp4'
    format_name = 'mp4v'
    

    video = cv2.VideoWriter(video_dir,cv2.VideoWriter_fourcc(*format_name),FRAME_RATE,(480*2 + 40, 272))

    for name in image_names:
        if not name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        
        image_path = os.path.join(file_dir, name)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Warning: Image at {image_path} could not be read.")
            continue

        video.write(img)
    
    video.release()
    


    
    # if os.path.exists(zero_img_dir):
    #     img = cv2.imread(zero_img_dir)
    # else:
    #     return
    # video = cv2.VideoWriter(video_dir,cv2.VideoWriter_fourcc(*format_name),FRAME_RATE,(img.shape[1], img.shape[0]))
    # while True:
    #     img_dir = os.path.join(file_dir, str(i)+'.png')
    #     if os.path.exists(img_dir):
    #         img = cv2.imread(img_dir)
    #         #img = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
    #         if no_alpha:
    #             img = img.astype(np.float32)
    #             #print(img.shape)
    #             img/=255.00
    #             img = img[...,:3]*img[...,-1:] + (1.-img[...,-1:])
    #             #img = img[...,:3] + (1.-img[...,-1:])
    #             img = (255*np.clip(img,0,1)).astype(np.uint8)
    #         video.write(img)
    #         i += 1
    #     else:
    #         break
    # print(i)
    # video.release()
    # return

file_dir = '/home/lhy/code/garage/gaussian-splatting/output/video3/train/ours_30000/compose'
video_dir = '/home/lhy/code/garage/gaussian-splatting/output/video3/train/ours_30000/compose.mp4'
convert_file_to_video(file_dir, video_dir)