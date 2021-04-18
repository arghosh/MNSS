# MNSS PyTorch

This is an official repository of the paper [Skill-based Career Path Modeling and Recommendation](https://people.umass.edu/~andrewlan/papers/20bigdata-mnss.pdf) published in [IEEE Big Data 2020 Conference](https://bigdataieee.org/BigData2020/). 

## Enviroment Setup
Create a python enviroment with the provided requirements file and [miniconda](https://docs.conda.io/en/latest/miniconda.html):

```(bash)
conda create --name mnss --file requirements.txt
conda activate mnss
```

## Training
MNSS model is added in `mnss.py` and NSS and NEMO model are added in `baseline.py`. The training command is:
```(bash)
python train.py --mode <model_name: mnss, nss, or nemo>
```
The monotonic GRU module is defined in `monotonic_gru.py`.

## Datasets
Linkedin dataset is retreived from [Kaggle](https://www.kaggle.com/linkedindata/linkedin-crawled-profiles-dataset) and Indeed dataset is collected from [Datastock](https://datastock.shop/download-indeed-job-resume-dataset/). 
Linkedin dataset is not available anymore in [Kaggle](https://www.kaggle.com/linkedindata/linkedin-crawled-profiles-dataset). Unfortunately, due to agreement with [Kaggle](https://www.kaggle.com/), we cannot release the collected Linkedin dataset. 

We added a sample dataset `demo.json` and required mappers in the `data/demo.zip` file for running `train.py` (unzip it). 

## Citation
If you find this code useful in your research then please cite  
```(bash)
@inproceedings{ghosh2020skill,
  title={Skill-based Career Path Modeling and Recommendation},
  author={Ghosh, Aritra and Woolf, Beverly and Zilberstein, Shlomo and Lan, Andrew},
  booktitle={2020 IEEE International Conference on Big Data (Big Data)},
  pages={1156--1165},
  year={2020},
  organization={IEEE}

}
``` 

Contact:  Aritra Ghosh (aritraghosh.iem@gmail.com).