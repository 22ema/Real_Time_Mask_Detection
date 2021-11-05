# Real_Time_Mask_Detection

Implementation to Detect Mask in webcam.
Improve performance by adding datasets.

 - Yolor

## Performance

| Dataset | Test Size | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | batch1 throughput |
| :-- | :-: | :-: | :-: | :-: |
| **LMDD(Imbalance dataset)** | 1280 | **??%** | **??%** | ?? *fps* |
| **??** | 1280 | **??%** | **??%** | ?? *fps* |
| **??** | 1280 | **??%** | **??%** | ?? *fps* |



## Installation
 - python version : 3.6.9
 - pip3 install -r requirements.txt
 
## Inference
 - download wights and cfg file
 - save in model directory
 - python3 run.py
 
## Evaluate
 - download testdataset and annotation_path, weights, cfg file
 - save in dataset directory and model directory
 - python3 evaluate.py 

## References

<details><summary> <b>Expand</b> </summary>

* [https://github.com/WongKinYiu/yolor.git](https://github.com/WongKinYiu/yolor.git)
* [https://www.kaggle.com/andrewmvd/face-mask-detection](https://www.kaggle.com/andrewmvd/face-mask-detection)
* [https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset)

</details>
