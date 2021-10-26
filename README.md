# DriveGAN: Towards a Controllable High-Quality Neural Simulation

PyTorch code for DriveGAN

**DriveGAN: Towards a Controllable High-Quality Neural Simulation** \
[Seung Wook Kim](http://www.cs.toronto.edu/~seung/), [Jonah Philion](https://www.cs.toronto.edu/~jphilion/), [Antonio Torralba](http://web.mit.edu/torralba/www/), [Sanja Fidler](http://www.cs.toronto.edu/~fidler/)\
CVPR (oral), 2021 \
**[[Paper](https://arxiv.org/abs/2104.15060)] [[Project Page](https://nv-tlabs.github.io/DriveGAN/)]**

**Abstract:**
Realistic simulators are critical for training and verifying robotics systems. While most of the contemporary simulators are hand-crafted, a scaleable way to build simulators is to use machine learning to learn how the environment behaves in response to an action, directly from data. In this work, we aim to learn to simulate a dynamic environment directly in pixel-space, by watching unannotated sequences of frames and their associated action pairs. We introduce a novel high-quality neural simulator referred to as DriveGAN that achieves controllability by disentangling different components without supervision. In addition to steering controls, it also includes controls for sampling features of a scene, such as the weather as well as the location of non-player objects. Since DriveGAN is a fully differentiable simulator, it further allows for re-simulation of a given video sequence, offering an agent to drive through a recorded scene again, possibly taking different actions. We train DriveGAN on multiple datasets, including 160 hours of real-world driving data. We showcase that our approach greatly surpasses the performance of previous data-driven simulators, and allows for new features not explored before.

For business inquires, please contact researchinquiries@nvidia.com

For press and other inquireis, please contact Hector Marinez at hmarinez@nvidia.com

### Citation
- If you found this codebase useful in your research, please cite:
```
@inproceedings{kim2021drivegan,
  title={DriveGAN: Towards a Controllable High-Quality Neural Simulation},
  author={Kim, Seung Wook and Philion, Jonah and Torralba, Antonio and Fidler, Sanja},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5820--5829},
  year={2021}
}
```

## Environment Setup
This codebase is tested with Ubuntu 18.04 and python 3.6.9, but it most likely would work with other close python3 versions.
- Clone the repository
```
git clone https://github.com/nv-tlabs/DriveGAN_code.git
cd DriveGAN_code
```
- Install dependencies
```
pip install -r requirements.txt
```

## Data
We provide a dataset derived from Carla Simulator (https://carla.org/, https://github.com/carla-simulator/carla).
This dataset is distributed under [Creative Commons Attribution-NonCommercial 4.0 International Public LicenseCC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode)

All data are stored in the following link:
https://drive.google.com/drive/folders/1fGM6KVzBL9M-6r7058fqyVnNcHVnYoJ3?usp=sharing

# Training
## Stage 1 (VAE-GAN)
If you want to skip stage 1 training, go to the Stage 2 (Dynamics Engine) section.
For stage 1 training, download  {0-5}.tar.gz from the [link](https://drive.google.com/drive/folders/1fGM6KVzBL9M-6r7058fqyVnNcHVnYoJ3?usp=sharing) and extract. The extracted datasets have names starting with 6405 - change their name to data1 (for 0.tar.gz) to data6 (for 5.tar.gz).
```
cd DriveGAN_code/latent_decoder_model
mkdir img_data && cd img_data
tar -xvzf {0-5}.tar.gz
mv 6405x data{1-6}
```

Then, run
```
./scripts/train.sh ./img_data/data1,./img_data/data2,./img_data/data3,./img_data/data4,./img_data/data5,./img_data/data6
```
You can monitor training progress with tensorboard in the log_dir specified in train.sh

When validation loss converges, you can now encode the dataset with the learned model (located in log_dir from training)
```
./scripts/encode.sh ${path to saved model} 1 0 ./img_data/data1,./img_data/data2,./img_data/data3,./img_data/data4,./img_data/data5,./img_data/data6 ../encoded_data/data
```  


## Stage 2 (Dynamics Engine)
If you did not do Stage 1 training, download encoded_data.tar.gz and vaegan_iter210000.pt from [link](https://drive.google.com/drive/folders/1fGM6KVzBL9M-6r7058fqyVnNcHVnYoJ3?usp=sharing), and extract.
```
cd DriveGAN_code
mkdir encoded_data
tar -xvzf encoded_data.tar.gz -C encoded_data
```

Otherwise, run
```
cd DriveGAN_code
./scripts/train.sh encoded_data/data ${path to saved vae-gan model}
```

# Playing with trained model
If you want to skip training, download simulator_epoch1020.pt and vaegan_iter210000.pt from [link](https://drive.google.com/drive/folders/1fGM6KVzBL9M-6r7058fqyVnNcHVnYoJ3?usp=sharing).

To play with a trained model, run
```
./scripts/play/server.sh ${path to saved dynamics engine} ${port e.g. 8888} ${path to saved vae-gan model}
```
Now you can navigate to localhost:{port} on your browser (tested on Chrome) and play.

(Controls - 'w': speed up, 's': slow down, 'a': steer left, 'd': steer right)

There are also additional buttons for changing contents.
To sample a new scene, simply refresh the webpage.

Thanks to @virtualramblas, this [link](https://github.com/virtualramblas/DriveGAN_code/tree/master/notebooks) has a notebook for playing it in Colab. 


# License
Thie codebase and trained models are distributed under [Nvidia Source Code License](https://github.com/nv-tlabs/DriveGAN_code/blob/master/LICENSE) and the dataset is distributed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

Code for VAE-GAN is adapted from https://github.com/rosinality/stylegan2-pytorch ([License](https://github.com/nv-tlabs/DriveGAN_code/blob/master/LICENSE-ROSINALITY)).

Code for Lpips is imported from https://github.com/richzhang/PerceptualSimilarity ([License](https://github.com/nv-tlabs/DriveGAN_code/blob/master/LICENSE-LPIPS)).

StyleGAN custom ops are imported from https://github.com/NVlabs/stylegan2 ([License](https://github.com/nv-tlabs/DriveGAN_code/blob/master/LICENSE-NVIDIA)).

Interactive UI code uses http://www.semantic-ui.com/ ([License](https://github.com/Semantic-Org/Semantic-UI/blob/master/LICENSE.md)).
