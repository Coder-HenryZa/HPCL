## Source code for the paper "Sensing the Diversity of Rumors: Rumor Detection with Hierarchical Prototype Contrastive Learning"

### Requirements

Code developed and tested in Python 3.9 using PyTorch 1.10.2, faiss-gpu 1.7.4 and Torch-geometric 2.2.0. Please refer to their official websites for installation and setup.

Some major dependencies are as follows:

```
emoji==2.12.1
faiss==1.7.4
Jinja2==3.1.4
joblib==1.4.2
Markdown==3.6
markdown-it-py==3.0.0
matplotlib==3.6.3
opt-einsum==3.3.0
optree==0.11.0
packaging==24.0
pandas==2.2.2
pillow==10.3.0
protobuf==4.25.3
psutil==5.9.8
Pygments==2.18.0
PyMySQL==1.1.1
pyparsing==3.1.2
python-dateutil==2.9.0.post0
pytz==2024.1
requests==2.32.2
rich==13.7.1
scikit-learn==1.5.0
scipy==1.13.1
six==1.16.0
tqdm==4.66.4
typing_extensions==4.12.0
tzdata==2024.1
urllib3==2.2.1
zipp==3.18.2
```

### Datasets

Data of Twitter15 and Twitter16 social interaction graphs follows this paper:

Tian Bian, Xi Xiao, Tingyang Xu, Peilin Zhao, Wenbing Huang, Yu Rong, Junzhou Huang. Rumor Detectionon Social Media with Bi-Directional Graph Convolutional Networks. AAAI 2020.

The raw datasets can be respectively downloaded from https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0.

The historical tweet data was crawled by the Twitter Developer tool in about January 2022, before the strict crawling restrictions were in place.  The tool's URL is https://developer.twitter.com/en. If a user's historical tweets are empty, it is because they were not successfully crawled. The reasons for this could be that the account was deleted or had an insufficient number of tweets.

### Run

```
# Data pre-processing
python ./util/getInteractionGraph.py Twitter15
python ./util/getInteractionGraph.py Twitter16
python ./networks/getTwitterTokenize.py Twitter15
python ./networks/getTwitterTokenize.py Twitter16
# run
python HPCL_Run.py
```

### Citation

If you find this repository useful, please kindly consider citing the following paper:

```
@article{zheng2024sensing,
  title={Sensing the diversity of rumors: Rumor detection with hierarchical prototype contrastive learning},
  author={Zheng, Peng and Dou, Yong and Yan, Yeqing},
  journal={Information Processing \& Management},
  volume={61},
  number={6},
  pages={103832},
  year={2024},
  publisher={Elsevier}
}
```







