# Early-Exit-Models

 This repository is dedicated to self-learning about early exit models, including relevant code and documentation.  

| Survey Papers    | code              | Comments    |
|-------------|-------------------------|-------------|
| 1. [Split Computing and Early Exiting for Deep Learning Applications: Survey and Research Challenges](https://dl.acm.org/doi/pdf/10.1145/3527155) |       | Fundamental Survey Paper Section 4|
| 2. [Adaptive Inference through Early-Exit Networks: Design, Challenges and Directions](https://arxiv.org/pdf/2106.05022) |  | |
| 3. [Distributed Artificial Intelligence Empowered by End-Edge-Cloud Computing: A Survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9933792)||Section III-C-4|
| 4. [End-Edge-Cloud Collaborative Computing for Deep Learning: A Comprehensive Survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10508191)|  | Section III-B|
| 5.[Advancements in Accelerating Deep Neural Network Inference on AIoT Devices: A Survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10398463) |  | |
| 6.[Resource Management in Mobile Edge Computing: A Comprehensive Survey](https://dl.acm.org/doi/pdf/10.1145/3589639) | | |


| Papers       | code                    | Comments   | 
|-------------|-------------------------|-------------|
| 1. [BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks](https://arxiv.org/abs/1709.01686)| [Official Code](https://gitlab.com/kunglab/branchynet)<br>[code](https://github.com/gorakraj/earlyexit_onnx/tree/master/Networks/6.%20BranchyNet)      | Fundamental  Paper |
| 2.[Distributed Deep Neural Networks over the Cloud, the Edge and End Devices](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7979979) | [Offical Code](https://github.com/kunglab/ddnn) | Follow up work, node-edge-cloud setting|
| 3.[Multi-Scale Dense Networks for Resource Efficient Image Classification](https://arxiv.org/pdf/1703.09844) | [Official Code](https://github.com/gaohuang/MSDNet)<br>[pytorch](https://github.com/kalviny/MSDNet-PyTorch) | MSDNet(ICLR)|
| 4.[Branchy-GNN: a Device-Edge Co-Inference Framework for Efficient Point Cloud Processing](https://arxiv.org/abs/2011.02422) | [Offical Code](https://github.com/shaojiawei07/Branchy-GNN) | GNN|
| 5.[EdgeKE: An On-Demand Deep Learning IoT System for Cognitive Big Data on Industrial Edge Devices](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9294146) | [Offical Code](https://github.com/fangvv/EdgeKE) | knowledge distillation, early exit to meet latency or accuracy requirements|
| 6.[SPINN: Synergistic Progressive Inference of Neural Networks over Device and Cloud](https://arxiv.org/pdf/2008.06402) | []() | run-time scheduler(Mobicom)|
| 7.[FlexDNN: Input-Adaptive On-Device Deep Learning for Efficient Mobile Vision](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9355785) | []() | |
| 8.[Boomerang: On-demand cooperative deep neural network inference for edge intelligence on the industrial internet of things](https://ieeexplore.ieee.org/abstract/document/8863733) | | |
| 9.[Early-exit deep neural networks for distorted images: providing an efficient edge offloading](https://arxiv.org/pdf/2108.09343) | [Offical Code](https://github.com/pachecobeto95/distortion_robust_dnns_with_early_exit) |early-exit DNN with expert branches  |
| 10.[FrameExit: Conditional Early Exiting for Efficient Video Recognition](https://arxiv.org/pdf/2104.13400) | [Offical Code](https://github.com/Qualcomm-AI-research/FrameExit) | gating module (CVPR)|
| 11.[BERxiT: Early exiting for BERT with better fine-tuning and extension to regression](https://aclanthology.org/2021.eacl-main.8.pdf) | [Offical Code](https://github.com/castorini/berxit) | (ACL)|
| 12.[A lightweight collaborative deep neural network for the mobile web in edge cloud](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9286558) | []() | Binary neural network branch |
| 13.[A Lightweight Collaborative Recognition System with Binary Convolutional Neural Network for Mobile Web Augmented Reality](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8884895) | []() | Binary neural network branch |
| 14.[DNN Inference Acceleration with Partitioning and Early Exiting in Edge Computing](https://dl.acm.org/doi/10.1007/978-3-030-85928-2_37) | []() | |
| 15.[Edge Intelligence: On-Demand Deep Learning Model Co-Inference with Device-Edge Synergy (Edgent)](https://arxiv.org/pdf/1806.07840) <br>[Edge AI: On-Demand Accelerating Deep Neural Network Inference via Edge Computing](https://arxiv.org/pdf/1910.05316) | []() | partitions DNN computation between mobile and edge server based on the available bandwidth|
| 16.[Improved Techniques for Training Adaptive Deep Networks](https://arxiv.org/pdf/1908.06294) | [Offical Code](https://github.com/kalviny/IMTA) | (ICCV)|
| 17.[Learning Anytime Predictions in Neural Networks via Adaptive Loss Balancing](https://arxiv.org/pdf/1708.06832) | []() | (AAAI)|
| 18.[Learning to Stop While Learning to Predict](https://arxiv.org/pdf/2006.05082) | [Offical Code](https://github.com/xinshi-chen/l2stop) | (ICML)|
| 19.[DeepAdapter: A Collaborative Deep Learning Framework for the Mobile Web Using Context-Aware Network Pruning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9155379) | []() | Follow up work of Edgent, online inference|
| 20.[Branching in Deep Networks for Fast Inference](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9054209) | []() | |
| 21.[Accelerating on-device DNN inference during service outage through scheduling early exit](https://www.sciencedirect.com/science/article/pii/S0140366420318818) | []() | |
| 22.[Learning Early Exit for Deep Neural Network Inference on Mobile Devices through Multi-Armed Bandits](https://ieeexplore.ieee.org/document/9499356) | []() | |
| 23.[Cloudedge-based lightweight temporal convolutional networks for remaining useful life prediction in IIoT](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9137209) | []() | two scale prediction |
| 24.[Predictive Exit: Prediction of Fine-Grained Early Exits for Computation- and Energy-Efficient Inference](https://arxiv.org/pdf/2206.04685) | []() | (AAAI)|
| 25.[DeeCap: Dynamic Early Exiting for Efficient Image Captioning](https://ieeexplore.ieee.org/document/9879601) | [Offical Code](https://github.com/feizc/DeeCap) | |
| 26.[A Simple Hash-Based Early Exiting Approach For Language Understanding and Generation](https://arxiv.org/pdf/2203.01670) | [Offical Code](https://github.com/txsun1997/HashEE) | (ACL) |
| 27.[It's always personal: Using Early Exits for Efficient On-Device CNN Personalisation](https://arxiv.org/abs/2102.01393) | []() | |
| 28.[Class-specific early exit design methodology for convolutional neural networks](https://www.sciencedirect.com/science/article/pii/S1568494621002398#b22) | []() | |
| 29.[Federated Learning for Cooperative Inference Systems: The Case of Early Exit Networks](https://arxiv.org/abs/2405.04249) |       | Cooperative Inference Systems settings |
| 30.[Dual Dynamic Inference: Enabling More Efficient, Adaptive, and Controllable Deep Inference](https://arxiv.org/pdf/1907.04523) | []() |Channel with Early Exit|
| 31.[Multi-Exit Semantic Segmentation Networks](https://arxiv.org/pdf/2106.03527) | []() | (ECCV)|
| 32.[Multi-Exit DNN Inference Acceleration Based on Multi-Dimensional Optimization for Edge Intelligence](https://ieeexplore.ieee.org/document/9769868) | []() | a contextual bandit learning that learns the optimal partition point|
| 33.[Accelerating on-device DNN inference during service outage through scheduling early exit](https://www.sciencedirect.com/science/article/pii/S0140366420318818) | []() | |
| 34.[Autodidactic Neurosurgeon: Collaborative Deep Inference for Mobile Edge Intelligence via Online Learning](https://arxiv.org/pdf/2102.02638) | []() | |
| 35.[Towards Edge Computing Using Early-Exit Convolutional Neural Networks](https://www.mdpi.com/2078-2489/12/10/431) | []() |  MobiletNetV2 with early exits|
| 36.[Resource-Constrained Edge AI with Early Exit Prediction](https://arxiv.org/pdf/2206.07269) | []() | |
| 37.[Temporal Decisions: Leveraging Temporal Correlation for Efficient Decisions in Early Exit Neural Networks](https://arxiv.org/html/2403.07958v1) | []() | |
| 38.[Efficient Post-Training Augmentation for Adaptive Inference in Heterogeneous and Distributed IoT Environments](https://arxiv.org/pdf/2403.07957) | []() | |
| 39.[DyCE: Dynamic Configurable Exiting for Deep Learning Compression and Scaling](https://arxiv.org/html/2403.01695v1) | []() | |
| 40.[ClassyNet: Class-Aware Early-Exit Neural Networks for Edge Devices](https://ieeexplore.ieee.org/abstract/document/10365527) | []() | |
| 41.[ENASFL: A Federated Neural Architecture Search Scheme for Heterogeneous Deep Models in Distributed Edge Computing Systems](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10366825) | []() | |
| 42.[Adaptive Early Exiting for Collaborative Inference over Noisy Wireless Channels](https://arxiv.org/pdf/2311.18098) | []() | |
| 43.[EdgeFM: Leveraging Foundation Model for Open-set Learning on the Edge](https://yanzhenyu.com/assets/pdf/EdgeFM-SenSys23.pdf) | []() | |
| 44.[Channel-Adaptive Early Exiting using Reinforcement Learning for Multivariate Time Series Classification](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10459930) | []() | |
| 45.[SplitEE: Early Exit in Deep Neural Networks with Split Computing](https://arxiv.org/abs/2309.09195) | [Offical Code](https://github.com/Div290/SplitEE/blob/main/README.md) | |
| 46.[Branchy Deep Learning Based Real-Time Defect Detection Under Edge-Cloud Fusion Architecture](https://ieeexplore.ieee.org/abstract/document/10149362) | []() | |
| 47.[Joint multi-user DNN partitioning and task offloading in mobile edge computing](https://www.sciencedirect.com/science/article/pii/S1570870523000768) | []() | |
| 48.[Resource-aware Deployment of Dynamic DNNs over Multi-tiered Interconnected Systems](https://arxiv.org/abs/2404.08060) | []() | |
| 49.[Edge Computing with Early Exiting for Adaptive Inference in Mobile Autonomous Systems](https://iris.polito.it/retrieve/e943f733-907f-4746-bbc1-3952a7ce945e/a495-angelucci%20final.pdf) | []() | |


# Repos on Early Exit
| Papers   | code               | Comments   | 
|-------------|-------------------------|-------------|
|  | [Repo](https://github.com/falcon-xu/early-exit-papers?tab=readme-ov-file) | |
|  | [Repo](https://github.com/txsun1997/awesome-early-exiting) | |

# AI on Edge 
| Papers   | code               | Comments    |
|-------------|-------------------------|-------------|
| Enabling AI on Edges: Techniques, Applications and Challenges | [Offical Code](https://github.com/wangxb96/Awesome-AI-on-the-Edge) | |
| [Green Edge AI: A Contemporary Survey](https://arxiv.org/pdf/2312.00333) | []() | |

![Early Exit --- Fig. 7 in 1st survey paper](Y_Matsubara_et_al.png)
**Early Exit inference model from Fig.7 in Split Computing and Early Exiting for Deep Learning Applications: Survey and Research Challenges, YOSHITOMO MATSUBARA and MARCO LEVORATO, University of California, Irvine, USA, FRANCESCO RESTUCCIA, Northeastern University, USA**
