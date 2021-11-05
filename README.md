# Real_Time_Mask_Detection

Implementation to Detect Mask in webcam.

 - Yolor

## Performance

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | batch1 throughput | batch32 inference |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| **??** | 1280 | **??%** | **??%** | **??%** | ?? *fps* | ?.? *ms* |
| **??** | 1280 | **??%** | **??%** | **??%** | ?? *fps* | ??.? *ms* |
| **??** | 1280 | **??%** | **??%** | **??%** | ?? *fps* | ??.? *ms* |
|  |  |  |  |  |  |  |


## Installation
 - python version : 3.6.9
 - pip install -r requirements.txt
 
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


</details>