# TEGCN_on_Hololens2


## Introduction
This repo is for Temporally Enhanced Graph Convolutional Network for Hand Tracking from an Egocentric Camera, Woojin Cho, et al.


## Setup
The following is the list of requirements for the unity project:

- Unity 2020.3.21f1 (LTS)* -> Unity 2022.3.60f1 (LTS)
- MRTK 2.8.3.0

And for the python project:
- Python v3.8+
- CUDA 11.7
- Pytorch v1.13.1+
  

- [Client](https://github.com/UVR-WJCHO/TEGCN_on_Hololens2/tree/main/Client) 
	- Open Unity projects in `WiseUIAppUnity`
	- ... (TBD. Check initial MRTK setting)
	
	
- [Server](https://github.com/UVR-WJCHO/TEGCN_on_Hololens2/tree/main/Server) 

	- Install the dependencies
	```
	pip install -r Server/requirements_cuda_v11_7.txt
	```	

	- Download pretrained model and Mano data (updated : 25/04/14)
	```	https://www.dropbox.com/scl/fi/mgtnhommqvrvm2exjbxjx/SAR_AGCN4_cross_wBGaug_extraTrue_resnet34_s0_Epochs50.zip?rlkey=pgxx00s6efc3jswutzessyafl&st=2jwhfpmy&dl=0
	https://www.dropbox.com/scl/fi/60hzlehmd74e2c3xo2pxz/mano.zip?rlkey=mrxkbn9yl06zmop6ml6n1ofsy&dl=0
	```

	- Locate the file at 
	```
	Server/handtracker/checkpoint/[model_name]/checkpoint.pth
	Server/handtracker/mano_data/mano
	```


## Build (Unity project on HoloLens2)

1. Open WiseUIAppUnity  in Unity.
2. In MRTK Project Configurator, select "Show Setting"
3. In Project Settings install XR Plugin Management if it is not.
4. In XR Plugin Management, switch to Universar Windows Platform setting, then tick "Windows Mixed Reality"
 
5. In the Project tab, open `Scenes/HoloLens2 PV Camera Test.unity`.
6. Select MixedRealityToolkit Gameobject in the Hierarchy. In the Inspector, change the mixed reality configuration profile to `New MixtedRealityToolkitConfigurationProfile`
7. Go to Build Settings, switch target platform to UWP.
8. Hopefully, there is no error in the console. Go to Build Settings, change Target Device to HoloLens, Architecture to ARM64. Build the Unity project in a new folder (e.g. 'Build' folder).

6. Save the changes. Open `Build/WiseUIAppUnity.sln`. Change the build type to Release/Master-ARM64-Device(or Remote Machine). Build - Deploy.


## Run
- [Client](https://github.com/UVR-WJCHO/TEGCN_on_Hololens2/tree/main/Client)  
	1. Run `WiseUIApp` in HoloLens2.
	2. Allow all permission requests.

- [Server](https://github.com/UVR-WJCHO/TEGCN_on_Hololens2/tree/main/Server) 
```
activate [virtualenv]
cd ./Server
python main.py
```


## Reference
```
@article{cho2024temporally,
  title={Temporally enhanced graph convolutional network for hand tracking from an egocentric camera},
  author={Cho, Woojin and Ha, Taewook and Jeon, Ikbeom and Jeon, Jinwoo and Kim, Tae-Kyun and Woo, Woontack},
  journal={Virtual Reality},
  volume={28},
  number={3},
  pages={1--18},
  year={2024},
  publisher={Springer}
}
```

## Acknowledgement
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2019-0-01270, and RS-2024-00397663).