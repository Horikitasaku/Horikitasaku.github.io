---
title: IceCube - Neutrinos in Deep Ice
description: The top 3% of all participating teams globally. First Silver Medal. My story in the competition.
toc: true
authors:
  - Horikita Saku
tags:
  - Astronomy
  - Neutrinos
  - IceCube
  - Deep Learning
categories:
  - Astronomy
series:
  - Neutrinos
date: '2023-10-26'
lastmod: '2023-10-26'
featuredImage: images/41185938.gif
draft: false
weight: 1
---
## Foreword 

I am writing this article in the hope of sharing insights from my personal journey in the competition, accompanied by a brief introduction to the techniques employed. 

For those seeking a more comprehensive technical understanding, I highly recommend delving into the outstanding paper authored by Habib Bukhari, Dipam Chakraborty, Philipp Eller, et al. [[8]](https://arxiv.org/abs/2310.15674). Their work provides an excellent and detailed technical explanation.

## Introduction

*One of the most abundant particles in the universe is the neutrino. While similar to an electron, the nearly massless and electrically neutral neutrinos have fundamental properties that make them difficult to detect. Yet, to gather enough information to probe the most violent astrophysical sources, scientists must estimate the direction of neutrino events. If algorithms could be made considerably faster and more accurate, it would allow for more neutrino events to be analyzed, possibly even in real-time and dramatically increase the chance to identify cosmic neutrino sources. Rapid detection could enable networks of telescopes worldwide to search for more transient phenomena.*

*Researchers have developed multiple approaches over the past ten years to reconstruct neutrino events. However, problems arise as existing solutions are far from perfect. They're either fast but inaccurate or more accurate at the price of huge computational costs.*

*The IceCube Neutrino Observatory is the first detector of its kind, encompassing a cubic kilometer of ice and designed to search for the nearly massless neutrinos. An international group of scientists is responsible for the scientific research that makes up the IceCube Collaboration.* [[1]](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice)

<div style="text-align: center;">
    <img src="../../../images/icecube.png" alt="IceCube"/>
    <p style="margin-top: 1px;">Figure1 IceCube</p>
</div>

---

Icecube detects neutrinos primarily by detecting the Cherenkov light. About Cherenkov light, it is the electromagnetic radiation emitted when charged particles pass through a dielectric at a speed greater than the phase velocity of light in the medium (the speed at which the wave front travels through the medium). I think the simple analogy to it is the sonic boom but Cerenkov light is a shock wave of light. So when neutrinos and nuclei react, they produce charged secondary particles, which in turn produce Cerenkov light. Then Icecube's Digital Optical module (DOM) can detect the Cerenkov light, telling us that a neutrino has passed through.

Therefore, in the competition, we use the the energy of pulsed light, the time the detector detected the light, and the sensor coordinates to indirectly predict the neutrino and its path. The target is zenith Angle and azimuth Angle.

### Possible difficulty

Detecting low-energy neutrinos has always been a big problem. Therefore, the possible difficulties of this competition are mainly due to:

1. Icecube is designed to observe neutrinos with energies of about one-tenth of a TeV. Although the DeepCore subdetector extends this range to 50 GeV, allowing the detection of low-energy neutrinos, there are still neutrinos with even lower energy.
2. Muons, produced by the interaction of cosmic rays with the Earth's atmosphere, can outnumber neutrino signals in IceCube. Approximately 1,000,000 muons are detected for every neutrino seen in IceCube.[[5](https://masterclass.icecube.wisc.edu/en/learn/detecting-neutrinos)]

### The Least Square

Least squares is the original method[[2](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/discussion/381747),[3](https://user-web.icecube.wisc.edu/~mhuennefeld/DNNreco/wikipage/material/MasterThesis.pdf),[4](https://inspirehep.net/literature/1409209)]. In this approach, the core idea is that if the pulse propagated in an omni-directional manner due to a neutrino collision, the time-weighted omni-directional properties should cancel each other out, leaving only the properties of the neutrino’s direction of travel.

So we replacing the charge with the mass to obtain the "center of charge" equivalent to the center of mass. 

In this way, we can use the method same to the original least squares to get two points and draw a straight line from the earliest point to the latest point.

Let $𝒗_𝒊$ be the x,y, and z coordinates of the observed value, and $𝒕𝟎_𝒊 (𝒕𝟏_𝒊)$ be the time weight, then we can simply get:

$$
v_0 = \frac{\sum(v_i \cdot charge_i \cdot t0_i)}{\sum(charge_i \cdot t0_i)}
$$
$$
v_1 = \frac{\sum(v_i \cdot charge_i \cdot t1_i)}{\sum(charge_i \cdot t1_i)}
$$

then $v_1-v_0$ is the direction vector of the neutrino, and its opposite is the direction of the source of the neutrino, that is, the value to be predicted.

Because of these properties, least squares regression is a suitable method for approximation of neutrino’s trajectory.

However, there are still many problems with this approach. The points used to perform the algorithm may not be strictly symmetric, and there may be pulse events that the detector missed and did not detect.

So, the score is not great, but the idea is still very interesting.

## DynEdge

We transitioned to utilizing Graph Neural Networks (GNN) in our approach, specifically employing ***DynEdge*** from Graphnet for the task of neutrino reconstruction and classification. A pivotal technology in this model is the *EdgeConv* operator, crucial for convolution on point cloud images. The main structure of *EdgeConv* is illustrated in Figure 2.

<div style="text-align: center;">
    <img src="../../../images/Graphnet.png" alt="Graphnet"/>
    <p style="margin-top: 1px;">Figure2 Structure of Graphnet</p>
</div>

The mathematical representation indicates that for each node 𝑛 𝑗 with node features 𝑥 𝑗 , the operator convolves 𝑥 𝑗 within the local neighborhood of 𝑛 𝑗 as:

$$
x_j = \sum_{i=1}^{N_{neighbors}} MLP(x_j - x_i)
$$

Here, $x_j$ represents the convolved node features of $n_j$, and the Multilayer Perceptron (MLP) takes as input the unconvolved node features of $n_j$ and the pairwise difference between the unconvolved node features of $n_j$ and its $i$-th neighbor. [[4](https://inspirehep.net/literature/1409209), [8](https://arxiv.org/abs/2310.15674)]

In essence, *EdgeConv* receives node features and their differences from neighbors, convolving them on the graph. Notably, its "convolution kernel" is determined by the number of links, meaning the "kernel" is defined by the edge, not its location. As depicted in the lower right corner of Figure 2, this step allows *DynEdge* to link arbitrary nodes in each convolution step, providing enhanced flexibility and a more comprehensive inclusion of intrinsic features between points, beyond just their coordinates.

Moreover, experimental results from the original paper demonstrate that *DynEdge* excels in performance, particularly for low-energy neutrinos, outperforming other contemporary methods. Consequently, *DynEdge* became the focal point of our efforts in this stage.

Despite these advancements, achieving an optimal score remains elusive, and my current ranking is still quite distant from the desired outcome.

## Switch to RNN

In the third stage, I, or most of the kagglers, switched to RNN.

In retrospect, this seems like a natural idea, because neutrinos pass through the detector in a time series. 

### Transform regression task into classification task

The idea comes from [[6](https://www.kaggle.com/code/seungmoklee/3-lstms-with-data-picking-and-shifting)].

This is a very interesting and important step in the LSTM solution.  The ingenious approach involves partitioning the zenith angle and azimuth angle into 24 segments each, encoding them using base 24, and transform the regression task into a classification tasks. Following this, the predicted values seamlessly map back to the regions encoded with 24.

This method greatly improves the score. We also tried to change the classification method at the end of the competition, but nothing worked, which is a pity.

### Different structures and ensembles

There's not a lot to explain. I simply designed a variety of LSTM and GRU combination structures, and weighted them to average. But in the final public solution, I learned an interesting way to use Xgboost for ensemble, which I hadn't considered at all before. [[7](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/discussion/402888)]

### ! Distinguish atmospheric muons !

At the end of the game for a month, my teammate [Enzo](https://www.kaggle.com/neomaoro) joined. It was at this stage that I found our greatest improvement.

<div class="image-container">
  <img src="../../../images/Muon_event.jpg" alt="Muon_event">
  <p class="image-description">(a) A muon entering the detector and leaving only a small amount of energy behind.</p>
</div>
<div class="image-container">
  <img src="../../../images/Neutrino_event.jpg" alt="Neurino_event">
  <p class="image-description">(b) A cascading burst event, with most of the energy remaining in the detector</p>
</div>
<div style="text-align: center;">
    <p style="margin-top: 1px;">Figure3 Examples of neutrino event topologies in IceCube from [<a href="https://arxiv.org/abs/1311.5238">11</a>]. Each panel is a schematic view of the detector, with each photomultiplier represented by a sphere whose volume is proportional to the collected charge. The smaller upper panels show projections of the detector along its z, x, and y axes, respectively.[<a href="https://iopscience.iop.org/article/10.1088/1748-0221/9/03/P03009">9</a>]</p>
</div>

Every event contains neutrinos, but there are some events where neutrinos are hard to detect. In Figure 3, the left side shows muons escaping the detector, of which only a small amount of energy is left in the detector, while the right side shows a cascade of neutrino events, of which most of the energy is left in the detector. The right side is very good for us, but it is likely that we cannot get such a perfect event on the right side because of the muons. [[9](https://iopscience.iop.org/article/10.1088/1748-0221/9/03/P03009)]**Therefore, for events that are difficult to reconstruct, we can focus on distinguishing atmospheric muons.**

In the context of separating atmospheric muons, a key strategy involves reconstructing their trajectories to ascertain whether the muons exited the Earth or were directed towards it. Given the inherent inability of muons to penetrate the Earth, the logic dictates that muons arriving from the Earth must be neutrinos. This principle allows for the exclusion of mesons from the atmosphere based on straightforward geometric considerations.

In addition, as shown in Figure 4, IceCube features zones with significant dust deposition, characterized by intense light absorption. Events occurring in these regions pose a considerable challenge for reconstruction and detection. [[10](https://www.science.org/doi/abs/10.1126/science.1242856)]

<div style="text-align: center;">
    <img src="../../../images/veto-zone.png" alt="vetozone"/>
    <p style="margin-top: 1px;">Veto-Zone</p>
</div>

After realizing this, we introduced the ***veto-zone***. In the *veto-region*, we try to exclude muons and dust deposition regions, in order to obtain better quality pulse.

Then We get two ways of filtering events.

-  **Filter A.** This filter involves event filtration based on angle and rejection zone criteria. The **reconstruction difficulty** of event is then computed, taking into account factors such as the horizontal direction, upward and downward considerations, veto-zone, and weights.
-  **Filter B.** In this filter (also the original filter), events are filtered based on a time window. The effective time period is determined by considering the time when the pulse is at its strongest and the speed of light. The rank(or quality) of a pulse is subsequently calculated, considering the weight of the pulse size. The optimal pulse is selected based on these considerations.

### Fitting -- Apply the filters

Experiment 1

In this initial experiment, Filter A incorporates physical properties, while Filter B is a straightforward statistical filter. My initial assumption was that events filtered by Filter A would exhibit higher quality compared to Filter B. Consequently, I utilized Filter B data for pre-training and Filter A data for fine-tuning. Surprisingly, the results were disheartening, leading to a sense of frustration, and I contemplated abandoning this approach.
However, a turning point emerged with Experiment 2.

Experiment 2

Upon closer inspection, I realized that I had misapplied one of the model's structures, leading to the omission of training for one of the models. Given the time constraints nearing the competition's end, retraining was deemed impractical. Instead, I loaded a model initially trained on Filter A data and fine-tuned it using Filter B data—essentially reversing the sequence from Experiment 1.

Contrary to my expectations, this approach yielded breakthrough results, surpassing my best cross-validation scores. This methodology was applied across all my models, propelling me into the top 3% of participants.

After competition, I observed that some of the top solutions also employed a two-stage training approach. Notably, these models were trained on low and high pulses. In contrast, our solution innovatively leverages physical properties, dividing the training into two stages, marking a distinctive approach that appears to be less commonly adopted.

## Final

IceCube competition definitely made me feel the fascination of science, like Prof. Francis Halzen said, '...what science is all about is not finding what you expect to find, is to finding something you didn't have any idea you were going to find, is to find the unexpected. '

## Reference

[1] Ashley Chow, Lukas Heinrich, Philipp Eller, Rasmus Ørsøe, Sohier Dane. (2023). IceCube - Neutrinos in Deep Ice. Kaggle. https://kaggle.com/competitions/icecube-neutrinos-in-deep-ice

[2] Mirco Hunnefield Masters Thesis, Online Reconstruction of Muon-Neutrino Events in IceCube using Deep Learning Techniques

[3] Schatto, K. Stacked searches for high-energy neutrinos from blazars with IceCube. (Mainz U., 2014).

[4] R. Abbasi, M. Ackermann, J. Adams, et al., “Graph neural networks for low-energy event classification & reconstruction in icecube,” Journal of Instrumentation, vol. 17, no. 11, P11003, 2022.

[5] Schatto, Kai. *Stacked searches for high-energy neutrinos from blazars with IceCube*. Diss. Mainz, Univ., Diss., 2014, 2014.

[6] *3 LSTMs; with Data Picking and Shifting | Kaggle*. (n.d.). Retrieved 25 January 2024, from https://www.kaggle.com/code/seungmoklee/3-lstms-with-data-picking-and-shifting

[7] *IceCube—Neutrinos in Deep Ice*. (n.d.). Retrieved 25 January 2024, from https://kaggle.com/competitions/icecube-neutrinos-in-deep-ice

[8] Bukhari, Habib, et al. "IceCube--Neutrinos in Deep Ice The Top 3 Solutions from the Public Kaggle Competition." *arXiv preprint arXiv:2310.15674* (2023).

[9] Aartsen, Mark G., et al. "Energy reconstruction methods in the IceCube neutrino telescope." *Journal of Instrumentation* 9.03 (2014): P03009.

[10] IceCube Collaboration*. "Evidence for high-energy extraterrestrial neutrinos at the IceCube detector." *Science* 342.6161 (2013): 1242856.

[11] Aartsen, M. G. "Evidence for high-energy extraterrestrial neutrinos at the icecube detector. Science 342." (2013).
<style>
  .image-container {
    display: inline-block;
    text-align: center;
    width: 49%;
  }

  .image-container img {
    max-width: 100%;
    height: auto;
  }

  .image-description {
    text-align: center;
  }
</style>