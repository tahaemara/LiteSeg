# LiteSeg
This the official implementation of "LiteSeg: A Litewiegth ConvNet for Semantic Segmentation"




## Installation
### Main Dependencies:
- Pytorch 0.4.1
- OpenCV 3.4.2
- pyyaml 3.13
- [LightNet](https://gitlab.com/tahaemara/lightnet.git)

Inorder to use this code you must install Anaconda and then apply the following steps:
+ Create the environment from the environment.yml file:

```
conda env create -f environment.yml
```
+ Activate liteseg environment

```
source activate liteseg
```

+ Install [LightNet](https://gitlab.com/tahaemara/lightnet.git) fork to be able to use Darknet weights 

```
git clone https://gitlab.com/tahaemara/lightnet.git

cd lightnet/

pip install -r requirements.txt
```
##### Note
> In case of have any problem while installing the dependencies you can igonre the message. Then activate liteseg environment and start to use the code. If any error message appears in a form of *ModuleNotFoundError: No module named 'xxxx'*, you can just search google how to install package xxxx with conda or pip.


## Training
#### Dataset
Before start training, download cityscapes dataset from [here](https://www.cityscapes-dataset.com) after regestering to the site. You need to download fine data (files leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip) and coarse data (leftImg8bit_trainextra.zip and gtCoarse.zip).
#### Training paramteres
[training.yaml](https://github.com/tahaemara/LiteSeg/blob/master/config/training.yaml) contains parameters needed for training as:
+ DATASET_FINE, path to fine dataset folder. The folder (cityscapes) must follow this pattern

    ``` 
    /path/to/cityscapes/leftImg8bit_trainvaltest/leftImg8bit
                                                            /train		
                                                            /val 	  
                                                            /test 
    /path/to/cityscapes/gtFine_trainvaltest/gtFine 
                                                  /train		
                                                  /val 	  
                                                  /test 
                                                  
   ```
   
+ DATASET_COARSE, path to coarse dataset folder. The folder (CityscapesExtra) must follow this pattern 

```
/path/to/CityscapesExtra/leftImg8bit
                                    /train		
                                    /val 	 
/path/to/CityscapesExtra/gtCoarse/gtCoarse
                                          /train		
                                          /val 	                                                   
```
+ PRETRAINED_SHUFFLENET, PRETRAINED_MOBILENET, and PRETRAINED_DarkNET19 parameters define locations of pre-trained backbone networks which can be downloaded from these links [Shufflenet](https://github.com/ericsun99/ShuffleNet-1g8-Pytorch/raw/master/ShuffleNet_1g8_Top1_67.408_Top5_87.258.pth.tar), [Mobilenet](https://github.com/ericsun99/MobileNet-V2-Pytorch/raw/master/mobilenetv2_Top1_71.806_Top2_90.410.pth.tar), and [Darknet19](https://pjreddie.com/media/files/darknet19_448.weights)

#### Start training:
In order to train the network with a specific backbone network and get and replicate the paper result you must train network on coarse data first, and then fine-tune the network with fine data.

+ To train on coarse data set **USING_COARSE to True** and run the training via 

``` python train_ms.py --backbone_network darknet ```

+ After that you can fine-tune the network using fine data but first you must set **USING_COARSE to False** and then run the training by passing name of backbone network and the path to the pretrained model file from the previous trainig -coarse- via 

```  python train_ms.py --backbone_network darknet --model_path_coarse ./pretrained_models/liteseg-darknet-cityscapes.pth ```

##### Note
You have an option to train network on fine data directley bust after setting **USING_COARSE to False** and commenting lines [107](https://github.com/tahaemara/LiteSeg/blob/b9f2b40cec88d63428fa90ec6d58ef55122974ff/train.py#L107), 108, and 109.

## Sample results


<table><tbody><tr><td>Samples</td><td>
    <img src="https://github.com/tahaemara/LiteSeg/blob/master/samples/frankfurt_000000_000294_leftImg8bit.png?raw=true" alt="" data-canonical-src="https://github.com/tahaemara/LiteSeg/blob/master/samples/frankfurt_000000_000294_leftImg8bit.png?raw=true" width="250" height="150">  
      </td><td>
  <img src="https://github.com/tahaemara/LiteSeg/blob/master/samples/frankfurt_000000_001016_leftImg8bit.png?raw=true" alt="" data-canonical-src="https://github.com/tahaemara/LiteSeg/blob/master/samples/frankfurt_000000_001016_leftImg8bit.png?raw=true" width="250" height="150">
  </td></tr><tr><td>Ground truth</td><td>
     <img src="https://github.com/tahaemara/LiteSeg/blob/master/samples/ground_truth/frankfurt_000000_000294_gtFine_color.png?raw=true" alt="" data-canonical-src="https://github.com/tahaemara/LiteSeg/blob/master/samples/ground_truth/frankfurt_000000_000294_gtFine_color.png?raw=true" width="250" height="150">  
    </td><td>
     <img src="https://github.com/tahaemara/LiteSeg/blob/master/samples/ground_truth/frankfurt_000000_001016_gtFine_color.png?raw=true" alt="" data-canonical-src="https://github.com/tahaemara/LiteSeg/blob/master/samples/ground_truth/frankfurt_000000_001016_gtFine_color.png?raw=true" width="250" height="150">  
    </td></tr><tr><td>LiteSeg-Darknet19</td><td>
    <img src="https://github.com/tahaemara/LiteSeg/blob/master/samples/predictions/frankfurt_000000_000294_leftImg8bit_liteseg-darknet.png?raw=true" alt="" data-canonical-src="https://github.com/tahaemara/LiteSeg/blob/master/samples/predictions/frankfurt_000000_000294_leftImg8bit_liteseg-darknet.png?raw=true" width="250" height="150">   
    </td><td>
    <img src="https://github.com/tahaemara/LiteSeg/blob/master/samples/predictions/frankfurt_000000_001016_leftImg8bit_liteseg-darknet.png?raw=true" alt="" data-canonical-src="https://github.com/tahaemara/LiteSeg/blob/master/samples/predictions/frankfurt_000000_001016_leftImg8bit_liteseg-darknet.png?raw=true" width="250" height="150">   
    </td></tr><tr><td>LiteSeg-Mobilenet</td><td>
    <img src="https://github.com/tahaemara/LiteSeg/blob/master/samples/predictions/frankfurt_000000_000294_leftImg8bit_liteseg-mobilenet.png?raw=true" alt="" data-canonical-src="https://github.com/tahaemara/LiteSeg/blob/master/samples/predictions/frankfurt_000000_000294_leftImg8bit_liteseg-mobilenet.png?raw=true" width="250" height="150">  
    </td><td>
    <img src="https://github.com/tahaemara/LiteSeg/blob/master/samples/predictions/frankfurt_000000_001016_leftImg8bit_liteseg-mobilenet.png?raw=true" alt="" data-canonical-src="https://github.com/tahaemara/LiteSeg/blob/master/samples/predictions/frankfurt_000000_001016_leftImg8bit_liteseg-mobilenet.png?raw=true" width="250" height="150">  
    </td></tr><tr><td>LiteSeg-Shufflenet</td><td>
      <img src="https://github.com/tahaemara/LiteSeg/blob/master/samples/predictions/frankfurt_000000_000294_leftImg8bit_liteseg-shufflenet.png?raw=true" alt="" data-canonical-src="https://github.com/tahaemara/LiteSeg/blob/master/samples/predictions/frankfurt_000000_000294_leftImg8bit_liteseg-shufflenet.png?raw=true" width="250" height="150">  
    </td><td>
      <img src="https://github.com/tahaemara/LiteSeg/blob/master/samples/predictions/frankfurt_000000_001016_leftImg8bit_liteseg-shufflenet.png?raw=true" alt="" data-canonical-src="https://github.com/tahaemara/LiteSeg/blob/master/samples/predictions/frankfurt_000000_001016_leftImg8bit_liteseg-shufflenet.png?raw=true" width="250" height="150">  
    </td></tr><tr><td>ErfNet</td><td><img src="https://github.com/tahaemara/LiteSeg/blob/master/samples/erfnet_predictions/frankfurt_000000_000294_leftImg8bit_erfnet.png?raw=true" alt="" data-canonical-src="https://github.com/tahaemara/LiteSeg/blob/master/samples/erfnet_predictions/frankfurt_000000_000294_leftImg8bit_erfnet.png?raw=true" width="250" height="150"> </td><td><img src="https://github.com/tahaemara/LiteSeg/blob/master/samples/erfnet_predictions/frankfurt_000000_001016_leftImg8bit_erfnet.png?raw=true" alt="" data-canonical-src="https://github.com/tahaemara/LiteSeg/blob/master/samples/erfnet_predictions/frankfurt_000000_001016_leftImg8bit_erfnet.png?raw=true" width="250" height="150"> </td></tr><tr><td>ESPNET</td><td><img src="https://github.com/tahaemara/LiteSeg/blob/master/samples/espnet_predictions/c_frankfurt_000000_000294_leftImg8bit.png?raw=true" alt="" data-canonical-src="https://github.com/tahaemara/LiteSeg/blob/master/samples/espnet_predictions/c_frankfurt_000000_000294_leftImg8bit.png?raw=true" width="250" height="150"> </td><td><img src="https://github.com/tahaemara/LiteSeg/blob/master/samples/espnet_predictions/c_frankfurt_000000_001016_leftImg8bit.png?raw=true" alt="" data-canonical-src="https://github.com/tahaemara/LiteSeg/blob/master/samples/espnet_predictions/c_frankfurt_000000_001016_leftImg8bit.png?raw=true" width="250" height="150"> </td></tr></tbody></table>

# License

This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/
