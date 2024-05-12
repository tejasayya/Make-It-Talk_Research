# MakeItTalk: Speaker-Aware Talking-Head Animation



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chakrapaul/MakeItTalk-Project/blob/main/Collab_Makeittalk_Demo.ipynb) &nbsp;

Link to Original GitHub: [https://github.com/yzhou359/MakeItTalk](https://github.com/yzhou359/MakeItTalk)

Link to Google Colab Notebook with Instructions: [Google Colab](https://colab.research.google.com/github/chakrapaul/MakeItTalk-Project/blob/main/Collab_Makeittalk_Demo.ipynb)


##Easy Steps to Execute in Google Colab:
USe the [Google Colab](https://colab.research.google.com/github/chakrapaul/MakeItTalk-Project/blob/main/Collab_Makeittalk_Demo.ipynb) Link to run the inference code that will generate a lip-synced video.



## Requirements

| **Requirements** | **Tool Name**     | **URL**                                                       |
| ---------------------- | ----------------------- | ------------------------------------------------------------------- |
| Operating System       | Windows                 |                                                                     |
| Programming Language   | 3.7                     | [https://www.python.org/downloads/](https://www.python.org/downloads/) |
| Framework              | PyTorch version >= 1.10 | [https://pytorch.org/](https://pytorch.org/)                           |
| Environment Manager    | Anaconda                | https://www.anaconda.com/download/success                           |

### <center><span style="color: red;">⚠️WARNING⚠️</span></center>

<center><span style="color: red;">Before beginning this tutorial, ensure that you have approximately 6 GB of available space on your computer. This is necessary because there are several large files required for the tutorial, and having sufficient free space will facilitate the process.</span></center>

### MakeItTalk at A Glance

MakeItTalk is a technique that creates a talking head video by combining a still image of a person's face with an audio clip. This method synthesizes facial movements that correspond to the speech in the audio, making the person in the image appear as if they are speaking the words. Additionally, MakeItTalk adds facial expressions that match the content and tone of the speech, resulting in a more natural and expressive talking head video that is synchronized with the audio input.

[Video](https://youtu.be/sQgTcWMFuic)

![image](https://github.com/pooja0207k/MakeItTalk/blob/main/metadta/paul%20bhai.jpeg)

In this practical tutorial section, you'll be guided through the process of converting a facial photograph and an audio file into a cohesive video using the provided graphical user interface (GUI).

### Manual Installation


#### Environment Setup

#### 1. Open a Terminal or Anaconda Prompt on Windows

**For Windows users**
In the Anaconda prompt, create a new conda environment with a custom name (for example, makeittalk_env) and Python 3.7 installation by entering the following command.

```
conda create -n makeittalk_env python=3.7
```

The Conda manager will download the required packages and ask for confirmation with a prompt: **Proceed ([y]/n)?** Type **y** to confirm and proceed with the downloads and installations.

Then, activate the newly created environment (makeittalk_env or whatever you have used) by typing the following:

```
conda activate makeittalk_env
```

The activated environment will now be displayed within parentheses at the beginning of your command lines, showing your environment name, as demonstrated below:

```
(makeittalk_env) [Your Computer Name]:~$
```

- ffmpeg (https://ffmpeg.org/download.html)

To enable audio and video processing necessary for creating the talking-head animation, you'll need to install FFmpeg, an open-source package.
```
sudo apt-get install ffmpeg
```
or
You can install FFmpeg by following the guidelines outlined in this URL: [Here](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/)

- Python packages
```
pip install -r requirements.txt
pip install tensorboardX
```


## Pre-trained Models

Download the following pre-trained models to `examples/ckpt` folder for testing your own animation.

| Model |  Link to the model | 
| :-------------: | :---------------: |
| Voice Conversion  | [Link](https://drive.google.com/file/d/1ZiwPp_h62LtjU0DwpelLUoodKPR85K7x/view?usp=sharing)  |
| Speech Content Module  | [Link](https://drive.google.com/file/d/1r3bfEvTVl6pCNw5xwUhEglwDHjWtAqQp/view?usp=sharing)  |
| Speaker-aware Module  | [Link](https://drive.google.com/file/d/1rV0jkyDqPW-aDJcj7xSO6Zt1zSXqn1mu/view?usp=sharing)  |
| Image2Image Translation Module  | [Link](https://drive.google.com/file/d/1i2LJXKp-yWKIEEgJ7C6cE3_2NirfY_0a/view?usp=sharing)  |
| Non-photorealistic Warping (.exe)  | [Link](https://drive.google.com/file/d/1rlj0PAUMdX8TLuywsn6ds_G6L63nAu0P/view?usp=sharing)  |


#### 2. PyTorch Installation

MakeItTalk was trained using the PyTorch framework, so you'll need to install PyTorch by visiting its official website (https://pytorch.org/) and choosing the configuration that corresponds to your system. For example, below is a configuration tailored for a Windows OS with a GPU card that supports CUDA 12.1.

<img width="452" alt="image" src="https://github.com/pooja0207k/MakeItTalk/blob/main/metadta/Screenshot%202024-05-06%20172741.png">

However, to avoid clashes due to packages version change, it is highly recommended to use the following command to install PyTorch.

```
conda install pytorch==1.10.2 torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
```


Python Libraries
- numpy: For numerical operations.
- opencv-python: For image Recognition & processing and video input/output.
- matplotlib: Python plotting library for creating static, interactive, and animated visualizations in a wide variety of formats.
- torch: PyTorch deep learning framework.
- pickle: serializes Python objects.
- face_alignment: detects and aligns facial landmarks.
- openvino: OpenVINO toolkit for optimized inference.
- pyaudio: For audio processing.
- Pillow: For image manipulation.



#### 3. Get The Code to Interact with MakeItTalk

**Note:** If you are given the code from your supervisor, please skip this step, and depending on your operation system, copy either **MakeItTalk** directory to your preferred working directory.

In order to get the code and try using MakeItTalk via the GUI, go to this GitHub URL: https://github.com/chakrapaul/MakeItTalk-Project and Download **MakeItTalk-Project** repository by clicking **Code ⇾ Download ZIP** as shown in the picture below.

<img width="451" alt="image" src="https://github.com/pooja0207k/MakeItTalk/blob/main/metadta/Screenshot%202024-05-06%20172835.png">

After downloading, extract **MakeItTalk-Project-main.zip** and check what you have inside. At this point, you should be able to find 2 directories under the **MakeItTalk-Project-main** folder.

<img width="362" alt="image" src="https://github.com/chakrapaul/MakeItTalk-Project/blob/main/metadata/lkl%20repo%20dwn.png">

if you are using Windows select **MakeItTalk** directory, cut, and paste it to a directory that you want to work on, for example, **D:** drive, **C:** drive, or in a folder where anaconda environment is present and all the environment paths are set.

**Note:** feel free to remove the **MakeItTalk-Project-main** folder after cutting **MakeItTalk** to your working directory(Anaconda Directory).

#### 4. Python Packages

Next, through the terminal or Anaconda prompt opened earlier, navigate yourself to **MakeItTalk** that you just pasted it under your working directory using the command below.

```
cd [path to your working directory]/[replace this with MakeItTalk]
```

Under **MakeItTalk**, requirements.txt file is provided to help you install the packages that you need.

<img width="432" alt="image" src="https://github.com/chakrapaul/MakeItTalk-Project/blob/main/metadata/requirements.png">

Type the command below in the terminal or Anaconda prompt to install the python packages in the requirements.txt file.

```
pip install -r requirements.txt
```

#### 5. Download Weighting of Pre-trained Models

```
!gdown -O examples/ckpt/ckpt_autovc.pth https://drive.google.com/uc?id=1ZiwPp_h62LtjU0DwpelLUoodKPR85K7x
!gdown -O examples/ckpt/ckpt_content_branch.pth https://drive.google.com/uc?id=1r3bfEvTVl6pCNw5xwUhEglwDHjWtAqQp
!gdown -O examples/ckpt/ckpt_speaker_branch.pth https://drive.google.com/uc?id=1rV0jkyDqPW-aDJcj7xSO6Zt1zSXqn1mu
!gdown -O examples/ckpt/ckpt_116_i2i_comb.pth https://drive.google.com/uc?id=1i2LJXKp-yWKIEEgJ7C6cE3_2NirfY_0a
!gdown -O examples/dump/emb.pickle https://drive.google.com/uc?id=18-0CYl5E6ungS3H4rRSHjfYvvm-WwjTI
```

<img width="432" alt="image" src="https://github.com/chakrapaul/MakeItTalk-Project/blob/main/metadata/encoders.png">

#### 6. Choose your image from the Dropdown.

```
import ipywidgets as widgets
import glob
import matplotlib.pyplot as plt
print("Choose the image name to animate: (saved in folder 'examples/')")
img_list = glob.glob1('examples', '*.jpg')
img_list.sort()
img_list = [item.split('.')[0] for item in img_list]
default_head_name = widgets.Dropdown(options=img_list, value='paint_boy')
def on_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        plt.imshow(plt.imread('examples/{}.jpg'.format(default_head_name.value)))
        plt.axis('off')
        plt.show()
default_head_name.observe(on_change)
display(default_head_name)
plt.imshow(plt.imread('examples/{}.jpg'.format(default_head_name.value)))
plt.axis('off')
plt.show()
```

####7. Seting up animation controllers

```
#@markdown # Animation Controllers
#@markdown Amplify the lip motion in horizontal direction
AMP_LIP_SHAPE_X = 2 #@param {type:"slider", min:0.5, max:5.0, step:0.1}

#@markdown Amplify the lip motion in vertical direction
AMP_LIP_SHAPE_Y = 2 #@param {type:"slider", min:0.5, max:5.0, step:0.1}

#@markdown Amplify the head pose motion (usually smaller than 1.0, put it to 0. for a static head pose)
AMP_HEAD_POSE_MOTION = 0.35 #@param {type:"slider", min:0.0, max:1.0, step:0.05}

#@markdown Add naive eye blink
ADD_NAIVE_EYE = True  #@param ["False", "True"] {type:"raw"}

#@markdown If your image has an opened mouth, put this as True, else False
CLOSE_INPUT_FACE_MOUTH = False  #@param ["False", "True"] {type:"raw"}          


#@markdown # Landmark Adjustment

#@markdown Adjust upper lip thickness (postive value means thicker)
UPPER_LIP_ADJUST = -1 #@param {type:"slider", min:-3.0, max:3.0, step:1.0}

#@markdown Adjust lower lip thickness (postive value means thicker)
LOWER_LIP_ADJUST = -1 #@param {type:"slider", min:-3.0, max:3.0, step:1.0}

#@markdown Adjust static lip width (in multipication)
LIP_WIDTH_ADJUST = 1.0 #@param {type:"slider", min:0.8, max:1.2, step:0.01}
```

### 8. Plotting landmarks on Image and Generating Facial Skeleton. Audio Signals maps to facial landmarks

```
import sys
sys.path.append("thirdparty/AdaptiveWingLoss")
import os, glob
import numpy as np
import cv2
import argparse
from src.approaches.train_image_translation import Image_translation_block
import torch
import pickle
import face_alignment
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
import shutil
import time
import util.utils as util
from scipy.signal import savgol_filter
from src.approaches.train_audio2landmark import Audio2landmark_model

sys.stdout = open(os.devnull, 'a')

parser = argparse.ArgumentParser()
parser.add_argument('--jpg', type=str, default='{}.jpg'.format(default_head_name.value))
parser.add_argument('--close_input_face_mouth', default=CLOSE_INPUT_FACE_MOUTH, action='store_true')
parser.add_argument('--load_AUTOVC_name', type=str, default='examples/ckpt/ckpt_autovc.pth')
parser.add_argument('--load_a2l_G_name', type=str, default='examples/ckpt/ckpt_speaker_branch.pth')
parser.add_argument('--load_a2l_C_name', type=str, default='examples/ckpt/ckpt_content_branch.pth') #ckpt_audio2landmark_c.pth')
parser.add_argument('--load_G_name', type=str, default='examples/ckpt/ckpt_116_i2i_comb.pth') #ckpt_image2image.pth') #ckpt_i2i_finetune_150.pth') #c
parser.add_argument('--amp_lip_x', type=float, default=AMP_LIP_SHAPE_X)
parser.add_argument('--amp_lip_y', type=float, default=AMP_LIP_SHAPE_Y)
parser.add_argument('--amp_pos', type=float, default=AMP_HEAD_POSE_MOTION)
parser.add_argument('--reuse_train_emb_list', type=str, nargs='+', default=[]) #  ['iWeklsXc0H8']) #['45hn7-LXDX8']) #['E_kmpT-EfOg']) #'iWeklsXc0H8', '29k8RtSUjE0', '45hn7-LXDX8',
parser.add_argument('--add_audio_in', default=False, action='store_true')
parser.add_argument('--comb_fan_awing', default=False, action='store_true')
parser.add_argument('--output_folder', type=str, default='examples')
parser.add_argument('--test_end2end', default=True, action='store_true')
parser.add_argument('--dump_dir', type=str, default='', help='')
parser.add_argument('--pos_dim', default=7, type=int)
parser.add_argument('--use_prior_net', default=True, action='store_true')
parser.add_argument('--transformer_d_model', default=32, type=int)
parser.add_argument('--transformer_N', default=2, type=int)
parser.add_argument('--transformer_heads', default=2, type=int)
parser.add_argument('--spk_emb_enc_size', default=16, type=int)
parser.add_argument('--init_content_encoder', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
parser.add_argument('--write', default=False, action='store_true')
parser.add_argument('--segment_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--emb_coef', default=3.0, type=float)
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
parser.add_argument('--use_11spk_only', default=False, action='store_true')
parser.add_argument('-f')
opt_parser = parser.parse_args()

img = cv2.imread('examples/' + opt_parser.jpg)
predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)
shapes = predictor.get_landmarks(img)
if (not shapes or len(shapes) != 1):
    print('Cannot detect face landmarks. Exit.')
    exit(-1)
shape_3d = shapes[0]
if(opt_parser.close_input_face_mouth):
    util.close_input_face_mouth(shape_3d)
shape_3d[48:, 0] = (shape_3d[48:, 0] - np.mean(shape_3d[48:, 0])) * LIP_WIDTH_ADJUST + np.mean(shape_3d[48:, 0]) # wider lips
shape_3d[49:54, 1] -= UPPER_LIP_ADJUST           # thinner upper lip
shape_3d[55:60, 1] += LOWER_LIP_ADJUST           # thinner lower lip
shape_3d[[37,38,43,44], 1] -=2.    # larger eyes
shape_3d[[40,41,46,47], 1] +=2.    # larger eyes
shape_3d, scale, shift = util.norm_input_face(shape_3d)

print("Loaded Image...", file=sys.stderr)

au_data = []
au_emb = []
ains = glob.glob1('examples', '*.wav')
ains = [item for item in ains if item is not 'tmp.wav']
ains.sort()
for ain in ains:
    os.system('ffmpeg -y -loglevel error -i examples/{} -ar 16000 examples/tmp.wav'.format(ain))
    shutil.copyfile('examples/tmp.wav', 'examples/{}'.format(ain))

    # au embedding
    from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
    me, ae = get_spk_emb('examples/{}'.format(ain))
    au_emb.append(me.reshape(-1))

    print('Processing audio file', ain)
    c = AutoVC_mel_Convertor('examples')

    au_data_i = c.convert_single_wav_to_autovc_input(audio_filename=os.path.join('examples', ain),
           autovc_model_path=opt_parser.load_AUTOVC_name)
    au_data += au_data_i
if(os.path.isfile('examples/tmp.wav')):
    os.remove('examples/tmp.wav')

print("Loaded audio...", file=sys.stderr)

# landmark fake placeholder
fl_data = []
rot_tran, rot_quat, anchor_t_shape = [], [], []
for au, info in au_data:
    au_length = au.shape[0]
    fl = np.zeros(shape=(au_length, 68 * 3))
    fl_data.append((fl, info))
    rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
    rot_quat.append(np.zeros(shape=(au_length, 4)))
    anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl.pickle'))):
    os.remove(os.path.join('examples', 'dump', 'random_val_fl.pickle'))
if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))):
    os.remove(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))
if(os.path.exists(os.path.join('examples', 'dump', 'random_val_au.pickle'))):
    os.remove(os.path.join('examples', 'dump', 'random_val_au.pickle'))
if (os.path.exists(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))):
    os.remove(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))

with open(os.path.join('examples', 'dump', 'random_val_fl.pickle'), 'wb') as fp:
    pickle.dump(fl_data, fp)
with open(os.path.join('examples', 'dump', 'random_val_au.pickle'), 'wb') as fp:
    pickle.dump(au_data, fp)
with open(os.path.join('examples', 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
    gaze = {'rot_trans':rot_tran, 'rot_quat':rot_quat, 'anchor_t_shape':anchor_t_shape}
    pickle.dump(gaze, fp)

model = Audio2landmark_model(opt_parser, jpg_shape=shape_3d)
if(len(opt_parser.reuse_train_emb_list) == 0):
    model.test(au_emb=au_emb)
else:
    model.test(au_emb=None)

print("Audio->Landmark...", file=sys.stderr)

fls = glob.glob1('examples', 'pred_fls_*.txt')
fls.sort()

for i in range(0,len(fls)):
    fl = np.loadtxt(os.path.join('examples', fls[i])).reshape((-1, 68,3))
    fl[:, :, 0:2] = -fl[:, :, 0:2]
    fl[:, :, 0:2] = fl[:, :, 0:2] / scale - shift

    if (ADD_NAIVE_EYE):
        fl = util.add_naive_eye(fl)

    # additional smooth
    fl = fl.reshape((-1, 204))
    fl[:, :48 * 3] = savgol_filter(fl[:, :48 * 3], 15, 3, axis=0)
    fl[:, 48*3:] = savgol_filter(fl[:, 48*3:], 5, 3, axis=0)
    fl = fl.reshape((-1, 68, 3))

    ''' STEP 6: Imag2image translation '''
    model = Image_translation_block(opt_parser, single_test=True)
    with torch.no_grad():
        model.single_test(jpg=img, fls=fl, filename=fls[i], prefix=opt_parser.jpg.split('.')[0])
        print('finish image2image gen')
    os.remove(os.path.join('examples', fls[i]))

    print("{} / {}: Landmark->Face...".format(i+1, len(fls)), file=sys.stderr)
print("Done!", file=sys.stderr)
```


### Animate Your Photo

```
from IPython.display import HTML
from base64 import b64encode

for ain in ains:
  OUTPUT_MP4_NAME = '{}_pred_fls_{}_audio_embed.mp4'.format(
    opt_parser.jpg.split('.')[0],
    ain.split('.')[0]
    )
  mp4 = open('examples/{}'.format(OUTPUT_MP4_NAME),'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

  print('Display animation: examples/{}'.format(OUTPUT_MP4_NAME), file=sys.stderr)
  display(HTML("""
  <video width=600 controls>
        <source src="%s" type="video/mp4">
  </video>
  """ % data_url))
```

result

<img width="432" alt="image" src="https://github.com/chakrapaul/MakeItTalk-Project/blob/main/metadata/animation.png">



## Or Simply download ipynb file [Here](https://github.com/chakrapaul/MakeItTalk-Project/blob/main/quick_demo_tdlr.ipynb) 
Open Jupyter Notebook from "makeittalk_env" Anaconda Environment and run This file cell by cell.





