#### Q1: Please put the imagenet result into mainpart of paper. add median number of queries for your ImageNet experiments in Table 14 as you did for CIFAR10 and MNIST in Tables 3, 4. Just the average number of queries is not sufficient in my opinion.

A1: Thank you for your comments. We have modified the structure of the article according to your suggestion, adding the Median item in Table 14.


#### Q2: Figure 5 is rather misleading because for the Square Attack in the second row we only see stripe initializaton without any sampled squares.

A2: Thank you for your suggestion. We have amend the figure in the Figure 5 of the main paper.


#### Q3: Could you elaborate on why fixed versions of existing attacks (e. g. Fixed-ZO-NGD) would be valuable baselines? The attacks were not designed that way and introducing this additional constraint seems to be an unclear step to me.

A3: In order to preserve the inherent characteristics of the baseline algorithm, we have made conservative changes to the existing algorithm (fixed areas), and in order to create as fair a comparison environment as possible, we also provide attack results for other areas in the appendix.

As far as we know, in recent years, there are fewer SOTA black-box regional attack algorithms that conform to the same scenario setting, so we change the SOTA global attack algorithm to fixed version. The good thing about this is, first of all, fixed versions of existing attacks provide a valuable baseline as they allow for a direct comparison under similar constraints. Secondly, it also effectively measures the attack performance of SOTA global attacks with the perturbation quantity limitation. As can be seen from Table 4, both the attack success rate and the query efficiency show a significant decline after the disturbance quantity limitation is added, thus highlighting the challenge of our paper.  Besides, from the results of the five positions provided in appendix, it can be seen that most of the best performance is concentrated in the center area, but it is still far behind us.

By comparing against these adapted baselines, the unique contributions and advantages of our method become more apparent. This includes demonstrating how automatic region selection can outperform traditional methods that do not focus on region perturbations. While these attacks were not originally designed with fixed constraints, adapting them in this manner helps to create a more comprehensive and robust evaluation framework. This approach is not intended to undermine the original design of these attacks but rather to provide a clearer context for evaluating the specific advancements our method offers.


#### Q4: Why would considering $\ell_\infty$ and $\ell_2$ metrics simultaneously e. g. in Table 2 be significant? 

A4: The goal of our method is to generate sparse and imperceptible perturbations. Therefore, we designed two constraints i.e. $\ell_\infty$ and $\ell^G_0$ , where $\ell^G_0$ is the group sparsity and $\ell_\infty$ is the constraint that the control perturbation is not perceptible. The $\ell_2$ norm is an metric to measure the imperceptibility of perturbations outside the constraint.

According to the original characteristics of the baseline, we divided the experiment into three sets(Global attack, Region-wise attack, Fixed attack).

* In the global attack, all the baseline algorithms have only $\ell_\infty$ constraints, so the sparsity and undetectability of perturbations are measured by $\ell_0$ and $\ell_2$ metrics, respectively.
* In the region-wise attack, all the baseline algorithms have only $\ell_0$ constraints, so the undetectability of perturbations is measured by $\ell_\infty$ and $\ell_2$ metrics.
* In the fixed attack, all the baseline algorithms have $\ell_{0+\infty}$  constraint, thus, it just need to compare the ASR and query efficiency.


#### Q5: If we wanted to minimize them simultaneously with the baseline attacks that you consider, we could include it as another term in the loss that they are trying to optimize. Have you considered such modifications to obtain better baselines?

[waiting for experiment result]
