# reply for PV33

Q1: Query effciency: I couldn’t find the comparison on how effcient is the current attack w.r.t the previous attacks. When permitted a high number of queries, the strength of most black-box attacks would increase, thus making it an unfair comparison in table 2. Second, it is critical to provide the number of queries vs attack strength to identify the pareto optimal curve of the current attack (currently the number of queries are set to fixed 10k, 40k - not sure why?). Similarly it is necessary to compare queries vs ASR plot with other attacks.

A1: Thanks for your comments. In the experimental scheme, we compare the performance of all algorithms with a limited query budget. Thus, we set different query budget for different dataset according to attack type and difficulty of the attack. As far as we know, the setting of this query budget is not strict, and can effectively compare the differences between different algorithms.

We set different query budget that is 5,000 , 10,000, 15,000, 20,000, 25,000 instead of fixed budget. Notwithstanding the inherent disparities in our respective problem, we divided three sets of experiments for different problem. First, we compared with global attack methods, i.e. Square attack, Parsimonious attack and ZO-NGD attack, with $\ell_\infty$ constraints in Fig.1. Thus, we set the same $\ell_\infty$ constraint for all algorithms is 0.05. Second, we compared with region-wise attack, i.e. Patch-RS, with a patch in Fig.2. We set the patch size of the square to be 80 in Patch-RS. Then,  the same $\ell_0$ constraint is 19200(80 $\times$ 80 $\times$ 3) in ours. Finally, we compared with fixed-version of global attack algorithms in Fig. 3. For all algorithms, we set the same double constraint, $\ell_\infty$ is 0.1, $\ell_0$ is 26820 for ImageNet dataset, Inceptionv3 model, and $\ell_\infty$ is 0.1, $\ell_0$ is 15052 for ImageNet dataset, vision transformer model. That is, for double constraint, we perturb only 10% of the pixels.

### Fig1 

From the Fig. 1, all algorithms have $\ell_\infty$ constraint 0.05. We can see that the success rate of the attack will decrease significantly if the query is less than 15,000 for high-resolution image attack tasks. When the query budget reaches 20,000, the success rate becomes stable.

### Fig2 

From the Fig. 2, all algorithms perturb up to 19200 of the pixels. Patch-RS is a heuristic attack method, it can be seen from the Fig. 2 inceptionv3 that Patch-RS performs very poorly under strict query budget. This is very different from how it works on the vision transformer. And our algorithm is based on optimization

### Fig3


Q2. Are there diminishing returns in attack success with higher resolution? While the proposed attack appears to be stronger and less perceptible than baselines of small resolution datasets (cifar10, mnist), the trend doesn't fully hold on ImageNet dataset (table 14 in appendix). Square attack [1] achieves equally high success rate and lower average queries.

A2: High-resolution images really improve the difficulty of attack a lot, especially in the case of double constraints. 

The problem addressed within the realms of the Square attack substantially diverges from the context elucidated within our paper. In our paper, our deliberations pivot around the conceptualization of regionally sparse attacks, a paradigm wherein we endow the capability to autonomously designate target regions for assault while concurrently regulating the number of regions to be subjected to such perturbation.   In stark contrast, the Square attack methodology is fundamentally engrossed in the task of imbuing perturbations across the entire expanse of an image, bereft of the nuanced capacity to pinpoint specific regions meriting subjection to attack.

From the table 14 in the appendix, if we perturb all the pixels like Square attack, we can achieve similar success rates and queries to Square attack. However,  from Tab.16 and Tab. 18 in the appendix, we can seen that Square attack doesn't do well with double constraint scenarios. 

Q3: Can authors clarify how the attack complexity behaves with experimental setup, i.e., input resolutions, size of neural networks, number of classes, etc?

Our experiments consisted of three datasets (MNIST, CIFAR10 and ImageNet) and four models. The dataset difficulty of the attack ranges from relatively simple MNIST to complex ImageNet.
