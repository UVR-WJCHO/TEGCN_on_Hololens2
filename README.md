# TEGCN_on_Hololens2


## Introduction
This repo is for Temporally Enhanced Graph Convolutional Network for Hand Tracking from an Egocentric Camera, Woojin Cho, et al.

## Update
+ 2024-3-19. Add base projects

## Features
- [x] Server - TEGCN
- [] Server - checkpoints
- [] Client - Unity project for Hololens2(Debug)
- [] Client - Unity project for Hololens2(Release)


## Install
+ Environment
    ```
    conda create -n tegcn python=3.9
    conda activate tegcn
    ```

+ Requirements
    ```
    pip install -r requirements.txt
    ```
	
+ You should accept [MANO LICENCE](https://mano.is.tue.mpg.de/license.html). Download MANO model from [official website](https://mano.is.tue.mpg.de/), then run
  ```
  ln -s /path/to/mano_v1_2/MANO_RIGHT.pkl Server/TEGCN/mano_data/mano/models/MANO_RIGHT.pkl
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