#### Q1: Some descriptions are confusing. The deduction in sec. 3.1 seems to be unnecessary, and the gradient estimation proposed in equ. (4) does not seem to differ from the standard gradient estimation approach.

A1: Thanks for your correction, we have fixed these errors in the text.

Thank you for your observations regarding the deduction in Section 3.1 and the gradient estimation method in Equation (4). We integrate this estimation technique with the Natural Evolutionary Strategies (NES). This integration is key to efficiently navigating the complex search space inherent in our proposed method. We will shorten the content of this section.


#### Q2: For computational cost and convergence, it only says high, medium and low. Are there any quantitative results to demonstrate it?

A2: In Table 1, we opted for qualitative descriptors (high, medium, low) to provide an initial, high-level comparative overview of our approach against others. This was intended to give readers a quick reference point for understanding the relative computational demands and convergence rates. We acknowledge that this presentation is flawed, and we will revise the table in the main text so that readers can better understand the different attack patterns. For cost analysis, we provide quantified results for reference in Appendix B.3.


#### Q3: It is not clear how the algorithm performs region selection. In algorithm 1, how the delta^0 is initialised? What is the initial perturbation group set G?

A3: We begin by dividing the input image into several potential regions based on predefined criteria. Our grouping method is to divide groups by a fixed sliding window. It can generate different perturbations by adjusting the window size and step size. In the section 3.3, we explain the process of the algorithm in detail, where the 9 line calculates the **DIS** of all groups according to the mask $I_G$. Then, k groups with the smallest **DIS** are selected by algorithm 2. The selection process is iterative and adaptive, meaning it can change in subsequent iterations based on the feedback received from the model's responses to previous perturbations.

In the version provided in the paper, we set the initial value of $\delta$ to be a uniformly distributed random disturbance. The initial perturbation group set G is empty.


#### Q4: Using the estimated gradient to perform black-box adversarial attacks is not new. Please refer to the SPSA attack [1].

A4: Thank you for pointing out the relevance of the SPSA attack in the context of our work. Our work contributes to the field by extending the concept of gradient estimation to more specific and challenging scenarios of adversarial attacks. So black box gradient attack method is not our main work and innovation point. Our method specifically addresses the challenge of region-wise adversarial attacks, focusing on improving efficiency and imperceptibility in this narrower domain. We propose specific algorithmic modifications and enhancements that are tailored to the unique requirements of region-wise attacks. This includes adaptations to handle structured sparsity constraints and the $\ell_\infty$ norm in a black-box setting. Our approach integrates the gradient estimation with Natural Evolutionary Strategies (NES), creating a novel synergy that enhances the efficacy of our attack method in terms of query efficiency and perturbation control.


#### Q5: In the experiment section. I am not sure if the comparison is fair, as different black-box attacks select different regions. Besides, the result shows that the proposed methods may actually require more queries as the median is significantly higher than other models. 

A5: According to the original characteristics of the baseline, we divided the experiment into three sets(Global attack, Region-wise attack, Fixed attack).

* In the global attack, all the baseline algorithms have only $\ell_\infty$ constraints, so the sparsity and undetectability of perturbations are measured by $\ell_0$ and $\ell_2$ metrics, respectively.
* In the region-wise attack, all the baseline algorithms have only $\ell_0$ constraints, so the undetectability of perturbations is measured by $\ell_\infty$ and $\ell_2$ metrics.
* In the fixed attack, all the baseline algorithms have $\ell_{0+\infty}$  constraint, thus, it just need to compare the ASR and query efficiency.

In the follow-up work, we adopted the variance reduction technology to effectively improve the query efficiency. We provide the latest results in the article.


#### Q6: Also, the authors used a simple CNN for the CIFAR10 dataset. It would be more convincing to evaluate pre-trained models from PyTorch model zoo or other resources.

A6: We evaluated the CIFAR10 dataset on two models, resnet18 and mobilenet_v2. The results of the evaluation are shown in the table below. In this evaluation, we also adopted variance reduction technology, and its query efficiency and attack success rate have been effectively improved.

**Table Performance on resnet18 and mobilenet_v2**


#### Q7: As the authors conducted a convergence analysis of the proposed attack. I am wondering if this can be further developed towards an adversarial verification method. Also, the robustness verification of adversarial patches has been done in [2]. I am also interested in the performance of the proposed attack on such a certified defence.

[To update]
