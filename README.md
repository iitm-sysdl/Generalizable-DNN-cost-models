# A Case for Generalizable DNN Cost Models for Mobile Devices [[Paper](https://vinodganesan.github.io/files/Generalizable_DNN_Cost_Models_IISWC_2020.pdf)][[Slides](https://slides.com/pratyushkumarpanda/deck/fullscreen?token=QQC26kZT)][[Videos](https://youtu.be/Mh1c-tNWRoc)]

```BibTex
@inproceedings{ganesan2020case,
  title={A Case for Generalizable DNN Cost Models for Mobile Devices},
  author={Ganesan, Vinod and Selvam, Surya and Sen, Sanchari and Kumar, Pratyush and Raghunathan, Anand},
  booktitle={2020 IEEE International Symposium on Workload Characterization (IISWC)},
  pages={169--180},
  year={2020},
  organization={IEEE}
}
```
## Desigining generalizable cost model for many devices
![](figures/overview.png)

## Characterisation Framework
![](figures/char_framework.png)

## Use case: Collaborative Workload Characterization
![](figures/Collaborative.png)

## Results
![](figures/Results.png)

## How to use our generalizable cost model?

* Generate random efficient networks and generate tflite models.

```bash
cd char_framework/net_generator/mobileOptGen
python MobileGen.py
```
* Deploy the generated tflite models on Android and measure the inference latency.

```bash
cd char_framework

```

* Learning a Cost Model
```bash
cd cost_model
python gen_cost_models.py
```

## Requirements

* Python 3+
* XGBoost
* Tensorflow 2+
* PyTorch
* ONNX
* torchprofile
* Android Studio (For measuring inference latency) 

## Slides
[![slides](figures/slides.png)](https://slides.com/pratyushkumarpanda/deck/fullscreen?token=QQC26kZT)

## Short Video
[![short video](https://img.youtube.com/vi/fOcvznSENUU/maxresdefault.jpg)](https://youtu.be/fOcvznSENUU)

## Full Video
[![full video](https://img.youtube.com/vi/Mh1c-tNWRoc/maxresdefault.jpg)](https://youtu.be/Mh1c-tNWRoc)
