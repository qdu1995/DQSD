# DQSD(TIP 2020)<br>
DQSD：Depth Quality Aware Salient Object Detection(Under review)<br>
The network structure is as follows:

# Requirment
---------
* tensorflow1.4<br>
* python3.6<br>
* keras2.1.2

# Preparation<br>

1.Clone code by clone git https://github.com/qdu1995/DQSD.git<br>
2.Download the official pre training weight Resnet or download initial model RGB stream,depth stream and DCA stream [link](https://pan.baidu.com/s/1E_eLNXN9l2mlpDxXdlohng)(89my) [link](https://pan.baidu.com/s/1wOXJD3mENKOgWok72ghYIQ)(3ftr) [link](https://pan.baidu.com/s/1SZL4EPqojn0LQEtzd_lgKQ)(x8uo)<br>
3.Download rgb-d dataset datasets in the folder of data for training or test.

for train:
---------
1.Put the train dataset into the train folder.<br>
2.Put the pretrained model into the 'model' folder.<br>
3.python train.py<br>


for test:
---------
1.Download testing data, put the test dataset into the test folder<br>
2.Download pretrained model [link](https://pan.baidu.com/s/1HpDPYcjIimngkwKpAJvICA)(5s17), put the model into the 'model' folder<br>
3.python demo.py<br>

Result:
---------
NJDU [link](https://pan.baidu.com/s/1ZdQeaYOVu1twxlstHwQs_g)(rtza) || NLPR [link](https://pan.baidu.com/s/1iDcEXuKq2FIoA6XIEeUcMg) (dfrz) || EDS [link](https://pan.baidu.com/s/1Udddumu1rvU2QKOFBuuTvQ)(pvgg) || LFSD [link](https://pan.baidu.com/s/1ty93u6NBQvHBKJErqy57hw)(kcp8) || SSD  [link](https://pan.baidu.com/s/1ymck12NEj6Px_sEyTOT3_Q)(py28) || STERE [link](https://pan.baidu.com/s/1Ph6nua51OBx9wy2qSFfKSg)(lw6w) || SIP [link](https://pan.baidu.com/s/1v-bTKFKLeljrE1zS_gYlfQ)(314e)

Evalution:
---------
Download the evaluation indicators [link](https://pan.baidu.com/s/1mk7KcpIOf_OXscVCW4kPuQ)(vhfc).Thanks for the authors (http://dpfan.net/) and [PDnet](https://github.com/cai199626/PDNet)<br>


If you think this work is helpful, please cite:
@article{

}<br>
qdu1995@163.com
