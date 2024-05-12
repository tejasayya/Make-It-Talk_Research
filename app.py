import streamlit as st
import os, glob
import numpy as np
import cv2
import argparse
from src.approaches.train_image_translation import Image_translation_block
import torch
import pickle
import face_alignment
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
import sys
import shutil
import time
import subprocess
import util.utils as util
from scipy.signal import savgol_filter
from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
from src.approaches.train_audio2landmark import Audio2landmark_model
import librosa


st.title("MakeItTalk Speaker Aware Head Animation");
uploaded_file = st.file_uploader("Upload an image")


# If user attempts to upload a file
if uploaded_file is not None:
    os.makedirs('img_dir', exist_ok=True)
    # Save the uploaded image to the temporary directory
    with open(os.path.join('img_dir', uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getvalue())

    # Get the path of the uploaded image
    image_path = os.path.join('img_dir', uploaded_file.name)

    # Display the uploaded image
    st.image(image_path)
    
uploaded_audio = st.file_uploader("Choose an audio file...", type=["mp3", "wav"])

if uploaded_audio is not None:
    audio_bytes = uploaded_audio.read()
    st.audio(audio_bytes, format='audio/wav')
    os.makedirs('audio_dir', exist_ok=True)
    with open(os.path.join('audio_dir', uploaded_audio.name), "wb") as f:
        f.write(uploaded_audio.read())

    
    
def setup_environment():
    st.write('Git clone project and install requirements...')
    os.system('git clone https://github.com/yzhou359/MakeItTalk.git')
    os.chdir('MakeItTalk/')
    os.system('pip install -r requirements.txt')
    os.system('pip install tensorboardX')
    os.makedirs('examples/dump', exist_ok=True)
    os.makedirs('examples/ckpt', exist_ok=True)
    os.system('pip install gdown')
    st.write('Done!')

def download_pretrained_models():
    st.write('Download pre-trained models...')
    os.system('gdown -O examples/ckpt/ckpt_autovc.pth https://drive.google.com/uc?id=1ZiwPp_h62LtjU0DwpelLUoodKPR85K7x')
    os.system('gdown -O examples/ckpt/ckpt_content_branch.pth https://drive.google.com/uc?id=1r3bfEvTVl6pCNw5xwUhEglwDHjWtAqQp')
    os.system('gdown -O examples/ckpt/ckpt_speaker_branch.pth https://drive.google.com/uc?id=1rV0jkyDqPW-aDJcj7xSO6Zt1zSXqn1mu')
    os.system('gdown -O examples/ckpt/ckpt_116_i2i_comb.pth https://drive.google.com/uc?id=1i2LJXKp-yWKIEEgJ7C6cE3_2NirfY_0a')
    os.system('gdown -O examples/dump/emb.pickle https://drive.google.com/uc?id=18-0CYl5E6ungS3H4rRSHjfYvvm-WwjTI')
    st.write('Done!') 
    
def librosa_installation():
    # Uninstall librosa
    subprocess.run(["pip", "uninstall", "librosa", "--yes"])

    # Install specific version of librosa
    subprocess.run(["pip", "install", "librosa==0.9.1"])
    

def animation_controllers():
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
     sys.path.append("thirdparty/AdaptiveWingLoss")
     st.echo()
     parser = argparse.ArgumentParser()
     parser.add_argument('--jpg', type=str, default={uploaded_file.name})
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
     opt_parser = parser.parse_args()

     img_bytes = uploaded_file.read()  # Read uploaded file bytes
     img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
     st.image(img, channels="BGR")

     shapes = detect_landmarks(img)
     if shapes is not None and len(shapes) == 1:
            shape_3d = shapes[0]
            # Perform adjustments
            if opt_parser.close_input_face_mouth:
                util.close_input_face_mouth(shape_3d)
            # Manipulate facial landmarks
            shape_3d[48:, 0] = (shape_3d[48:, 0] - np.mean(shape_3d[48:, 0])) * LIP_WIDTH_ADJUST + np.mean(shape_3d[48:, 0]) # wider lips
            shape_3d[49:54, 1] -= UPPER_LIP_ADJUST  # thinner upper lip
            shape_3d[55:60, 1] += LOWER_LIP_ADJUST  # thinner lower lip
            shape_3d[[37,38,43,44], 1] -= 2          # larger eyes
            shape_3d[[40,41,46,47], 1] += 2          # larger eyes

            shape_3d, scale, shift = util.norm_input_face(shape_3d)
            
            print("Loaded Image...", file=sys.stderr)
            
            au_data = []
            au_emb = []
            os.chdir('..')
            ains = glob.glob1('audio_dir', '*.wav')
            ains = [item for item in ains if item is not 'tmp.wav']
            ains.sort()
            for ain in ains:
                me, ae = get_spk_emb('audio_dir'.format(ain))
                au_emb.append(me.reshape(-1))

                print('Processing audio file', ain)
                c = AutoVC_mel_Convertor('audio_dir')

                au_data_i = c.convert_single_wav_to_autovc_input(audio_filename=os.path.join('audio_dir', ain),
                    autovc_model_path=opt_parser.load_AUTOVC_name)
                au_data += au_data_i
            
            # Placeholder for facial landmarks data
            fl_data = []
            rot_tran, rot_quat, anchor_t_shape = [], [], []
            
            # Create placeholder facial landmarks data
            for au, info in au_data:
                au_length = au.shape[0]
                fl = np.zeros(shape=(au_length, 68 * 3))
                fl_data.append((fl, info))
                rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
                rot_quat.append(np.zeros(shape=(au_length, 4)))
                anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))
                
            os.chdir('MakeItTalk')
            
            dump_dir = 'examples/dump'
            os.makedirs(dump_dir, exist_ok=True)  # Create dump directory if not exists

            if os.path.exists(os.path.join(dump_dir, 'random_val_fl.pickle')):
                os.remove(os.path.join(dump_dir, 'random_val_fl.pickle'))

            if os.path.exists(os.path.join(dump_dir, 'random_val_au.pickle')):
                os.remove(os.path.join(dump_dir, 'random_val_au.pickle'))

            if os.path.exists(os.path.join(dump_dir, 'random_val_gaze.pickle')):
                os.remove(os.path.join(dump_dir, 'random_val_gaze.pickle'))

            with open(os.path.join(dump_dir, 'random_val_fl.pickle'), 'wb') as fp:
                pickle.dump(fl_data, fp)

            with open(os.path.join(dump_dir, 'random_val_au.pickle'), 'wb') as fp:
                pickle.dump(au_data, fp)

            with open(os.path.join(dump_dir, 'random_val_gaze.pickle'), 'wb') as fp:
                gaze = {'rot_trans': rot_tran, 'rot_quat': rot_quat, 'anchor_t_shape': anchor_t_shape}
                pickle.dump(gaze, fp)

            # Run the audio-to-landmark model (replace this with your actual model execution logic)
            st.write("Running Audio to Landmark model...")

            # Instantiate the model (replace this with your actual model instantiation logic)
            model = Audio2landmark_model(opt_parser, jpg_shape=shape_3d)

            # Test the model with placeholder data (replace this with your actual model testing logic)
            if len(opt_parser.reuse_train_emb_list) == 0:
                model.test(au_emb=au_emb)
            else:
                model.test(au_emb=None)

            st.write("Audio to Landmark conversion completed.")
            
            fls = glob.glob1('examples', 'pred_fls_*.txt')
            fls.sort()
            
            for i in range(len(fls)):
                fl = np.loadtxt(os.path.join('examples', fls[i])).reshape((-1, 68, 3))
                fl[:, :, 0:2] = -fl[:, :, 0:2]
                fl[:, :, 0:2] = fl[:, :, 0:2] / scale - shift

                if ADD_NAIVE_EYE:
                    fl = util.add_naive_eye(fl)

                # Additional smoothing
                fl = fl.reshape((-1, 204))
                fl[:, :48 * 3] = savgol_filter(fl[:, :48 * 3], 15, 3, axis=0)
                fl[:, 48 * 3:] = savgol_filter(fl[:, 48 * 3:], 5, 3, axis=0)
                fl = fl.reshape((-1, 68, 3))

                # Image-to-image translation
                sys.path.append('src/approaches/train_image_translation.py')
                model = Image_translation_block(opt_parser, single_test=True)
                with torch.no_grad():
                    model.single_test(jpg=img, fls=fl, filename=fls[i], prefix=opt_parser.jpg.split('.')[0])

                # Remove processed file
                os.remove(os.path.join('examples', fls[i]))

                # Print progress
                st.write(f"{i+1} / {len(fls)}: Landmark->Face...")

            # Print completion message
            st.write("Done!")
            # visual_Animation();
            from base64 import b64encode
            from IPython.display import HTML
            for ain in ains:
                os.chdir("..")
                OUTPUT_MP4_NAME = '{}_pred_fls_{}_audio_embed.mp4'.format(opt_parser.jpg.split('.')[0], uploaded_file.name.split('.')[0])
                mp4 = open('audio_dir/{}'.format(OUTPUT_MP4_NAME), 'rb').read()
                data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
                os.chdir("MakeItTalk/")
                st.write('Display animation: examples/{}'.format(OUTPUT_MP4_NAME))
                st.video(data_url)
            
            
            
def visual_Animation():
    from base64 import b64encode
    from IPython.display import HTML
    OUTPUT_MP4_NAME = '{}_pred_fls_{}_audio_embed.mp4';
    mp4 = open('examples/{}'.format(OUTPUT_MP4_NAME), 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    st.write('Display animation: examples/{}'.format(OUTPUT_MP4_NAME))
    st.video(data_url)


    

def detect_landmarks(img):
    predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device='cpu', flip_input=True)
    shapes = predictor.get_landmarks(img)
    return shapes




def main():
    if st.button('Predict'):
            setup_environment()
            download_pretrained_models()
            librosa_installation()
            animation_controllers()

if __name__ == '__main__':
    main()
