# Multi-Sample Hypernetworks
The source code for the paper [Improving Pareto Front Learning via Multi-Sample Hypernetworks](https://arxiv.org/abs/2212.01130).

<p align="center"> 
    <img src="https://github.com/longhoangphi225/MultiSample-Hypernetworks/blob/main/.github/images/Screen%20Shot%202022-12-16%20at%2000.23.30.png" width="1000">
</p>  

# Run experiment
Please ```cd``` to the experiment folder and run
```
python trainer.py
```


# Citation
If our framework is useful for your research, please consider to cite the paper:
```
@article{Hoang_Le_Anh Tuan_Ngoc Thang_2023, title={Improving Pareto Front Learning via Multi-Sample Hypernetworks}, volume={37}, url={https://ojs.aaai.org/index.php/AAAI/article/view/25953}, DOI={10.1609/aaai.v37i7.25953}, abstractNote={Pareto Front Learning (PFL) was recently introduced as an effective approach to obtain a mapping function from a given trade-off vector to a solution on the Pareto front, which solves the multi-objective optimization (MOO) problem. Due to the inherent trade-off between conflicting objectives, PFL offers a flexible approach in many scenarios in which the decision makers can not specify the preference of one Pareto solution over another, and must switch between them depending on the situation. However, existing PFL methods ignore the relationship between the solutions during the optimization process, which hinders the quality of the obtained front. To overcome this issue, we propose a novel PFL framework namely PHN-HVI, which employs a hypernetwork to generate multiple solutions from a set of diverse trade-off preferences and enhance the quality of the Pareto front by maximizing the Hypervolume indicator defined by these solutions. The experimental results on several MOO machine learning tasks show that the proposed framework significantly outperforms the baselines in producing the trade-off Pareto front.}, number={7}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Hoang, Long P. and Le, Dung D. and Anh Tuan, Tran and Ngoc Thang, Tran}, year={2023}, month={Jun.}, pages={7875-7883} }
```
