## Efficient Style-Corpus Constrained Learning for Photorealistic Style Transfer

## Prerequisites
- Python3.7
- Pytorch 1.1


## Test
- Clone from github: 

    `git clone https://github.com/lixinghpu/SCCL.git`

    `cd SCCL`

- Download pre-trained models.

    Baidu：https://pan.baidu.com/s/19JENFU19r-KZx3poxPfzHQ

    mask： 6lv2



- Generate the output image.

    
    ```
    python test.py --content "content_path"  --style  "style_path"  --output_name "result_name"  --model_state_path  "pretrained_dir"
    ```

## Acknowledgement

　　Our implementation is highly inspired from the implementation of [AdaIN](https://github.com/irasin/Pytorch_AdaIN) and [WCT<sup>2</sup>](https://github.com/clovaai/WCT2).

## References

- [X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf)
- [Jaejun Yoo and Youngjung Uh. "Photorealistic Style Transfer via Wavelet Transforms." in ICCV, 2019.](https://arxiv.org/pdf/1903.09760.pdf)


### Citation
  
　　@article{

　　　　title={Efficient Style-Corpus Constrained Learning for Photorealistic Style Transfer},

　　　　author={Yingxu Qiao, Jiabao Cui, Fuxian Huang, Hongmin Liu*, Cuizhu Bao, Xi Li},

　　　　journal={IEEE Trans. On Image Processing},

　　　　year={2021}

　　　}
    

## Contact

　　Feel free to contact me if there is any question (qiaoyingxu@hpu.edu.cn).
