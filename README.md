# MakeItTalk Demo


<video  src="https://github.com/chakrapaul/MakeItTalk-Project/blob/main/metadata/pablo%20paul.mp4" type="video/mp4"> </video>


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

![image](https://github.com/pooja0207k/MakeItTalk/blob/main/metadta/paul%20bhai.jpeg)

In this practical tutorial section, you'll be guided through the process of converting a facial photograph and an audio file into a cohesive video using the provided graphical user interface (GUI).

### Installation Procedure

#### 1. Open a Terminal or Anaconda Prompt on Windows

#### 2. Environment Setup

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

#### 3. PyTorch Installation

MakeItTalk was trained using the PyTorch framework, so you'll need to install PyTorch by visiting its official website (https://pytorch.org/) and choosing the configuration that corresponds to your system. For example, below is a configuration tailored for a Windows OS with a GPU card that supports CUDA 12.1.

<img width="452" alt="image" src="https://github.com/pooja0207k/MakeItTalk/blob/main/metadta/Screenshot%202024-05-06%20172741.png">

However, to avoid clashes due to packages version change, it is highly recommended to use the following command to install PyTorch.

```
conda install pytorch==1.10.2 torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
```

#### 4. Multimedia Processing Package Installation

To enable audio and video processing necessary for creating the talking-head animation, you'll need to install FFmpeg, an open-source package.

For Windows users, you can install FFmpeg by following the guidelines outlined in this URL: https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/.

#### 5. Get The Code to Interact with MakeItTalk

**Note:** If you are given the code from your supervisor, please skip this step, and depending on your operation system, copy either **MakeItTalk** directory to your preferred working directory.

In order to get the code and try using MakeItTalk via the GUI, go to this GitHub URL: https://github.com/pooja0207k/MakeItTalk and Download **MakeItTalk** repository by clicking **Code ⇾ Download ZIP** as shown in the picture below.

<img width="451" alt="image" src="https://github.com/pooja0207k/MakeItTalk/blob/main/metadta/Screenshot%202024-05-06%20172835.png">

After downloading, extract **MakeItTalk-main.zip** and check what you have inside. At this point, you should be able to find 2 directories under the **MakeItTalk-main** folder.

<img width="362" alt="image" src="https://github.com/PhurinutR/MakeItTalk_Demo/assets/106614460/d08a1827-b8ee-4762-898d-c707aa6a0ce7">

if you are using Windows select **MakeItTalk** directory, cut, and paste it to a directory that you want to work on, for example, **D:** drive, **C:** drive, or others.

**Note:** feel free to remove the **MakeItTalk-main** folder after cutting **MakeItTalk** to your working directory.

#### 6. Python Packages

Next, through the terminal or Anaconda prompt opened earlier, navigate yourself to **MakeItTalk** that you just pasted it under your working directory using the command below.

```
cd [path to your working directory]/[replace this with MakeItTalk_Windows]
```

Under **MakeItTalk**, requirements.txt file is provided to help you install the packages that you need.

<img width="432" alt="image" src="https://github.com/chakrapaul/MakeItTalk-Project/blob/main/metadata/requirements.png">

Type the command below in the terminal or Anaconda prompt to install the python packages in the requirements.txt file.

```
pip install -r requirements.txt
```

#### 7. Download Weighting of Pre-trained Models

**Note:** This is a confidencial folder, It contains main encoders built by authors. Due to copyright issues we cannot disclose these files in our repo. If you are given the code from your supervisor, please skip this step.

<img width="432" alt="image" src="https://github.com/chakrapaul/MakeItTalk-Project/blob/main/metadata/encoders.png">

### Animate Your Photo

result

<img width="432" alt="image" src="https://github.com/chakrapaul/MakeItTalk-Project/blob/main/metadata/animation.png">
