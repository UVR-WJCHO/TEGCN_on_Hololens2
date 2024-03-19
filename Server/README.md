## Installation
### Requirements
- Python-3.8
- CUDA 11.7
- requirements.txt

## Setup - HandTracker

- Download pretrained model (To be updated)
```
https://www.dropbox.com/s/frp7cil5eeizo5b/SAR_refineWeight_cross_v2_extraTrue_resnet34_Epochs50.zip?dl=0
```

- Locate the file at 
```
WISEUIServer/handtracker/checkpoint/[model_name]/checkpoint.pth
```

- Check the path of pretrained model in `WISEUIServer/handtracker/config.py`, `checkpoint` parameter

- run 
```
activate [virtualenv]
cd ./WISEUIServer
python main.py
```