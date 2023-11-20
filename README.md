# Pytorch Model 만들기

```cpp
│  main.py
│  vggnet-19.cfg
│
├─dataSet
│  │  customloaderexample.py
│  │  datasetloader.py
│  │  data_transformer.py
│  │  __init__.py
│  │
│  ├─imageSet
│  │      cloudy1.jpg
│  │      cloudy2.jpg
│  │      labels.csv
│  │      rain1.jpg
│  │      rain2.jpg
│  │      shine1.jpg
│  │      shine2.jpg
│  │      sunrise1.jpg
│  │      sunrise2.jpg
│  │
│  ├─showImage
│  │  │  utils.py
│  │  │
│  │  └─__pycache__
│  └─__pycache__
├─model
│  │  model.py
│  │  VGG.py
│  │  __Init__.py
│  │
│  └─__pycache__
├─pth
├─train
│  │  trainer.py
│  │  __init__.py
│  │
│  └─__pycache__
└─utils
    │  tensor.md
    │  tensor.py
    │  tools.py
    │  __init__.py
    │
    └─__pycache__
```

위와 같은 파일 구조를 가진다. 

`main.py` 에서 모델과 데이터를 Import로 불러와서 모델을 생성하고 데이터를 불러온다. 

- 모델은 다음과 같은 import 문인 `from model import VGG` 로 불러온다.
- 데이터는 다음과 같은 import 문인 `from dataSet import datasetloader` 로 불러온다.

# 데이터 로더 생성

데이터는  datasetloader에 미리 코드로 작성한 customloader를 통해서 데이터를 불러온다. 

```python
import os

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomImageDataSet(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        # check files Three channels
        if image.shape[0] != 3:
            return self[idx + 1]
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image, label = self.transform([image, label])
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
```

# Model

Model은 Model 파일 아래에 구조를 작성하고 cfg 파일로 불러올 수 있는 구조를 생성합니다. 

```python
def read_config(path):
    """ read config files """
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

    return lines

def parse_model_config(path):
    """ Parse your model of configuration files and return module defines """
    lines = read_config(path)
    module_configs = []

    for line in lines:
        if line.startswith('['):
            layer_name = line[1:-1].rstrip()
            if layer_name == "net":
                continue
            module_configs.append({})
            module_configs[-1]['type'] = layer_name

            if module_configs[-1]['type'] == 'convolutional':
                module_configs[-1]['batch_normalize'] = 0
        else:
            if layer_name == "net":
                continue
            key, value = line.split("=")
            value = value.strip()
            module_configs[-1][key.rstrip()] = value.strip()

    return module_configs
```

tools에 작성한 코드로 Config 파일을 불러옵니다. 

Config 파일을 바탕으로 모델의 아키텍쳐를 생성합니다. 

```python
def set_layer(self):
        """ set layer from configuration file """
        module_list = nn.ModuleList()
        in_channels = [self.input_image_channel]

        for idx, info in enumerate(self.config):
            modules = nn.Sequential()

            if info['type'] == "convolutional":
                filters = int(info['filters'])
                modules = self._add_conv2d_layer(idx, modules, info, in_channels[-1], True)
                in_channels.append(filters)
            elif info["type"] == "maxpool":
                modules.append(
                    nn.MaxPool2d(kernel_size=int(info["size"]), stride=int(info["stride"])))
            module_list.append(modules)
        return module_list
```

<aside>
💡 향후 추가되는 기능에 따라 추가될 예정입니다.

</aside>