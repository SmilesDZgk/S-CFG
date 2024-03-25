## Semantic-aware  Classifier-Free Guidance (S-CFG)
This is the pytorch implementation of the paper "Rethinking the Spatial Inconsistency in Classifier-Free Diffusion Guidance"

### Setup
Our code builds on the reuqirement of the offical [Stable Diffusion repository](https://huggingface.co/runwayml/stable-diffusion-v1-5) and  [DeepFloyd IF repository](https://huggingface.co/DeepFloyd/IF-I-M-v1.0) in Hugging Face. 



### Usage
To generate an image, you can simply run the script by 

> cd IF \
> python IF.py

or

>cd sd \
>python sd.py

The detailed settings can be found in the file `IF.py` and  `sd.py`, where the prompt, and seed can be specified.






### Reference
If you find our methods or code helpful, please kindly cite the paper: 
```angular2
@inproceedings{shen2024rethinking,
  title={Rethinking the Spatial Inconsistency in Classifier-Free Diffusion Guidancee},
  author={Shen, Dazhong and Song, Guanglu and Xue, Zeyue and Wang, Fu-Yun and Liu, Yu},
  booktitle={Proceedings of The IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  year={2024}
}
``` 
