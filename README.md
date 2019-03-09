# LiteSeg
This the official implementation of "LiteSeg: A Litewiegth ConvNet for Semantic Segmentation"




# Installation
Inorder to use this code you must install Anaconda and then apply the following steps:
+ Create the environment from the environment.yml file:

```
conda env create -f environment.yml
```

+ Install [LightNet](https://gitlab.com/tahaemara/lightnet.git) fork to be able to use Darknet weights 

```
git clone https://gitlab.com/tahaemara/lightnet.git

cd lightnet/

pip install -r requirements.txt
```

# Sample results


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

#License

This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/
