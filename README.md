# Benchmarking Multi-Scene Fire and Smoke Detection

<p align="center">
  <img src="./assets/logo_benchmark_FSD.png" align="center" width="100%">
  
 <p align="center">
  <a href="https://link.springer.com/chapter/10.1007/978-981-97-8795-1_14" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-PRCV%202025-1765A5?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2410.16631" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-Arxiv-FFD700?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://xiaoyihan6.github.io/FSD/" target='_blank'>
    <img src="https://img.shields.io/badge/Page-XiaoyiHan6/FSD-C43779?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/Data-Previous FSD Datasets-CD5C5C?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/Data-MS FSDB-8CC63E?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://github.com/XiaoyiHan6/MS-FSDB" target='_blank'>
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FXiaoyiHan6%2FMS-FSDB&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=true">
 </p>
  </a>
<p align="center">
  <font size=5><strong>Benchmarking Multi-Scene Fire and Smoke Detection</strong></font>
    <br>
        <a href="https://xiaoyihan6.github.io/">Xiaoyi Han</a>,
        <a href="https://tpcd.github.io/">Nan Pu</a>,
        <a href="https://person.zju.edu.cn/fengzunlei">Zunlei Feng</a>,<br>
        <a href="https://person.zju.edu.cn/beiyj">Yijun Bei</a>,
        <a href="https://person.zju.edu.cn/zhangqf">Qifei Zhang</a>,
        <a href="https://faculty.hfut.edu.cn/ChengLechao/zh_CN/index.htm">Lechao Cheng</a>,
        <a href="https://csaic.szcu.edu.cn/2023/0721/c3057a54298/page.htm">Liang Xue</a>,<br>
    <br>
  Zhejiang University & University of Trento & Hefei University of Technology & Suzhou City University
  <br>
  Accepted to PRCV 2025
  </p>
</p>

---
<p align="center">
Note
</p>
Dear Visitor,
&nbsp;&nbsp;Hello! If you would like to use our Unified FSD Datasets, please click on the Google Drive link and send a request for access to the shared folder. Thank you.
<br>
<br>
Best regards,
<br>
<br>
Xiaoyi Han


---
We will provide five public Fire and Smoke Detection (FSD) datasets along with our own MS-FSDB.
The Five public FSD datasets include:

:one: :fire: VisiFire

:pencil2: B. U. Toreyin, A. E. Cetin, Online detection of fire in video, in: 2007 IEEE
Conference on Computer Vision and Pattern Recognition, 2007, pp. 1–5. [doi:
10.1109/CVPR.2007.383442](https://sci-hub.se/10.1109/cvpr.2007.383442)

:two: :fire: FIRESENSE

:pencil2: K. Dimitropoulos, P. Barmpoutis, N. Grammalidis, Spatio-temporal flame model-
ing and dynamic texture analysis for automatic video-based fire detection, IEEE
Transactions on Circuits and Systems for Video Technology 25 (2) (2015) 339–351.
[doi:10.1109/TCSVT.2014.2339592](https://sci-hub.st/10.1109/tcsvt.2014.2339592).

:three: :fire: Furg-Fire-Dataset 

:pencil2: V. H¨uttner, C. R. Steffens, S. S. da Costa Botelho, First response fire combat: Deep
leaning based visible fire detection, in: 2017 Latin American Robotics Symposium
(LARS) and 2017 Brazilian Symposium on Robotics (SBR), IEEE, 2017, pp. 1–6.
[doi:10.1109/SBR-LARS-R.2017.8215312](https://sci-hub.st/10.1109/sbr-lars-r.2017.8215312).


:four: :fire: BowFire

:pencil2: D. Y. T. Chino, L. P. S. Avalhais, J. F. Rodrigues, A. J. M. Traina, Bowfire:
Detection of fire in still images by integrating pixel color and texture analysis, in:
2015 28th SIBGRAPI Conference on Graphics, Patterns and Images, 2015, pp.
95–102. [doi:10.1109/SIBGRAPI.2015.19](https://arxiv.org/abs/1506.03495).

:five: :fire: Fire-Smoke-Dataset

:pencil2: DeepQuestAI, Fire-smoke-dataset (2021). URL https://github.com/DeepQuestAI/Fire-Smoke-Dataset

---

The directory structure of all these FSD Datasets follows the layout below:(1.VisiFire, 2.FIRESENSE, 3.Furg-Fire-Dataset, 4.BowFire, 5.Fire-Smoke-Dataset, 6.MS-FSDB)

```
FSD Dataset
|
|--data--|--predefined_classes.txt (Fire and Smoke)
|
|--images--|--000001.jpg
|
|--labels--|--000001.xml
|
|--layout--|--train.txt
|          |--test.txt
|          |--minitrain.txt
|          |--minitest.txt
|          |...
```
##BibTeX   
---

@InProceedings{han2025prcv,
      author="Han, Xiaoyi and Pu, Nan and Feng, Zunlei and Bei, Yijun and Zhang, Qifei and Cheng, Lechao and Xue, Liang", 
      editor="Lin, Zhouchen and Cheng, Ming-Ming and He, Ran and Ubul, Kurban and Silamu, Wushouer and Zha, Hongbin and Zhou, Jie and Liu, Cheng-Lin", 
      title="Benchmarking Multi-Scene Fire and Smoke Detection",booktitle="Pattern Recognition and Computer Vision", 
      year="2025", publisher="Springer Nature Singapore", address="Singapore", pages="203--218", 
      abstract="The current irregularities in existing public Fire and Smoke Detection (FSD) datasets have become a bottleneck in the advancement of FSD technology. Upon in-depth analysis, we identify the core issue as the lack of standardized dataset construction, uniform evaluation systems, and clear performance benchmarks. To address this issue and drive innovation in FSD technology, we systematically gather diverse resources from public sources to create a more comprehensive and refined FSD benchmark. Additionally, recognizing the inadequate coverage of existing dataset scenes, we strategically expand scenes, relabel, and standardize existing public FSD datasets to ensure accuracy and consistency. We aim to establish a standardized, realistic, unified, and efficient FSD research platform that mirrors real-life scenes closely. Through our efforts, we aim to provide robust support for the breakthrough and development of FSD technology. The project is available at https://xiaoyihan6.github.io/FSD/.",
isbn="978-981-97-8795-1"}
---
---
**Note**: Could you please give me a "one-click triple support"🔥 ("**Star**"🚀,"**Fork**"🔖,"**Issues**"❓)<br>
