# WACV2023 - Multimodal Person ReID Using ML-MDA 
Pytorch code for the [Multimodal Data Augmentation for Visual-Infrared Person ReID
with Corrupted Data](https://openaccess.thecvf.com/content/WACV2023W/RWS/papers/Josi_Multimodal_Data_Augmentation_for_Visual-Infrared_Person_ReID_With_Corrupted_Data_WACVW_2023_paper.pdf) paper. The code was developped on [1] code basis.

| Datasets      | Models                                                                     |
|---------------|----------------------------------------------------------------------------|
| #RegDB        | [GoogleDrive](https://drive.google.com/drive/folders/1ZO4oFfsA1eXhthMVHVs00ZQxPhQGy_8u?usp=sharing)|
| #SYSU-MM01    | [GoogleDrive](https://drive.google.com/drive/folders/11XQsD2ZG07oARTZFgLFnF85w9s0-MJiZ?usp=sharing)|
| #ThermalWORLD | [GoogleDrive](https://drive.google.com/drive/folders/16ZRVWpkcNOiHVAWdHv0Qeaa2cXKpDZX1?usp=sharing)|

#### 1. Prepare dataset 

Datasets can be directly downloaded and used as ease. The dataset file location can be set through args.data_path.
**args.data_path=/path/to/datasets**

- (1) RegDB Dataset [2]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

    - Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1) on their website). 

    - A private download link can be requested via sending an email to mangye16@gmail.com. 
  
- (2) SYSU-MM01 Dataset [3]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

- (3) ThermalWORLD Dataset [4]: The ThermalWorld Dataset ReID Split dataset can be downloaded by writing to vl.kniaz@gosniias.ru [website](http://www.zefirus.org/articles/ee9462fb-befd-4679-9c26-acd551db8583/)

#### 2. Train

To train our multimodal concatenation model using ML-MDA fold 0 on RegDB dataset, run: 
```
python train.py --dataset=RegDB \
                --model=concatenation \ 
                --data_path=../Datasets \
                --models_path=../save_model \
                --reid=BtoB \
                --ML_MDA \
                --fold=0;
```

  - `--dataset`: which dataset: "SYSU", "RegDB", or "TWorld".

  - `--model`: which model: "unimodal", "concatenation", "LMBN" [5], or "transreid" [6].
  
  - `--data_path`: Location of the dataset.

  - `--models_path`: Storage location of the learned models.

  - `--reid`: "BtoB" for concatenation model, "VtoV" for others.
  
  - `--ML_MDA`: Training using our ML-MDA strategy.
  
  - `--fold`: Fold to train from 0 to 4. 

To fully learn a model (5 folds), and select the wanted learning parameters dynamically, run ``` bash train.sh``` file and tune the required parameters accordingly.

#### 3. Inference

To test our multimodal concatenation model on SYSU-MM01-C*, learned using ML-MDA on RegDB dataset, run similarly:
```
python test.py  --dataset=SYSU \
                --model=concatenation \
                --data_path=../Datasets \
                --models_path=../save_model \
                --reid=BtoB \
                --ML_MDA \
                --scenario_eval=C*;
```

  - `--dataset`: which dataset: "SYSU", "RegDB", or "TWorld".

  - `--model`: which model: "unimodal", "concatenation", "LMBN" [5], or "transreid" [6].
  
  - `--data_path`: Location of the dataset.

  - `--models_path`: Storage location of the learned models.

  - `--reid`: "BtoB" for concatenation model, "VtoV" for others.
  
  - `--ML_MDA`: Model trained using our ML-MDA strategy.
  
  - `--scenario_eval`: Evaluation type, "normal" for clean datasets, "C" for visible modality corrupted only, "C*" for both visible and thermal modalities corrupted. 

To test a model, and select the wanted inference parameters dynamically, run ``` bash test.sh``` and tune the required parameters accordingly.

Trained models can be downloaded for direct inference using the previous file command.

*Results may vary from the paper for evaluation done on corrupted test sets as transformations are randomly applied. 

#### 4. References

[1] Ye, M., Shen, J., Lin, G., Xiang, T., Shao, L., & Hoi, S. C. (2021). Deep learning for person re-identification: A survey and outlook. IEEE transactions on pattern analysis and machine intelligence, 44(6), 2872-2893.

[2] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

[3] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

[4] Kniaz, V. V., Knyaz, V. A., Hladuvka, J., Kropatsch, W. G., & Mizginov, V. (2018). Thermalgan: Multimodal color-to-thermal image translation for person re-identification in multispectral dataset. In Proceedings of the European Conference on Computer Vision (ECCV) Workshops (pp. 0-0).

[5] Herzog, F., Ji, X., Teepe, T., Hörmann, S., Gilg, J., & Rigoll, G. (2021, September). Lightweight multi-branch network for person re-identification. In 2021 IEEE International Conference on Image Processing (ICIP) (pp. 1129-1133). IEEE.

[6] He, S., Luo, H., Wang, P., Wang, F., Li, H., & Jiang, W. (2021). Transreid: Transformer-based object re-identification. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 15013-15022).

### Citation

If this work helped your research, please kindly cite our paper:
```
@article{josi2022multimodal,
  title={Multimodal Data Augmentation for Visual-Infrared Person ReID with Corrupted Data},
  author={Josi, Arthur and Alehdaghi, Mahdi and Cruz, Rafael MO and Granger, Eric},
  journal={arXiv preprint arXiv:2211.11925},
  year={2022}
}
```
