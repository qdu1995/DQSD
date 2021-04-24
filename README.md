# DQSD(TIP 2020)<br>
DQSD：Depth Quality Aware Salient Object Detection(Under review)<br>
The network structure is as follows:

# Requirment
---------
* tensorflow1.4<br>
* python3.6<br>

# Preparation<br>

1.Clone code by clone git https://github.com/qdu1995/DQSD.git<br>
2.Download the official pre training weight Resnet or download initial model [link](https://pan.baidu.com/s/1q1Aw7J2l6XCqIGGBBu9HHA)(jsf8)<br>
3.Download rgb-d dataset datasets in the folder of data for training or test.

for train:
---------
1.Put the train dataset into the train folder.<br>
2.Put the pretrained model into the 'model' folder.<br>
3.python train.py<br>


for test:
---------
1.Download testing data, put the test dataset into the test folder<br>
2.Download pretrained model [link](https://pan.baidu.com/s/1sBXwplyNeNlqUjL-QRvppQ)(1nrn), put the model into the 'model' folder<br>
3.python demo.py<br>

Result:
---------
data [link](https://pan.baidu.com/s/1s_7zyAp2qxz6EwLQ7CA-ww)(j4iz)

Evalution:
---------
Download the evaluation indicators [link](https://pan.baidu.com/s/1mk7KcpIOf_OXscVCW4kPuQ)(vhfc).Thanks for the authors (http://dpfan.net/) and [PDnet](https://github.com/cai199626/PDNet)<br>


If you think this work is helpful, please cite:
@article{

}<br>
qdu1995@163.com
