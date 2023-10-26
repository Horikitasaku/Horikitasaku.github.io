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
## Introduction
For the technical section, read more about Habib Bukhari, Dipam Chakraborty, Philipp Eller et al. Their papers are excellent. [8]

I'm here to tell more about my own story.

## Description

*One of the most abundant particles in the universe is the neutrino. While similar to an electron, the nearly massless and electrically neutral neutrinos have fundamental properties that make them difficult to detect. Yet, to gather enough information to probe the most violent astrophysical sources, scientists must estimate the direction of neutrino events. If algorithms could be made considerably faster and more accurate, it would allow for more neutrino events to be analyzed, possibly even in real-time and dramatically increase the chance to identify cosmic neutrino sources. Rapid detection could enable networks of telescopes worldwide to search for more transient phenomena.*

*Researchers have developed multiple approaches over the past ten years to reconstruct neutrino events. However, problems arise as existing solutions are far from perfect. They're either fast but inaccurate or more accurate at the price of huge computational costs.*

*The IceCube Neutrino Observatory is the first detector of its kind, encompassing a cubic kilometer of ice and designed to search for the nearly massless neutrinos. An international group of scientists is responsible for the scientific research that makes up the IceCube Collaboration.* [1]

<div style="text-align: center;">
    <img src="../../../images/icecube.png" alt="IceCube"/>
    <p style="margin-top: 1px;">IceCube</p>
</div>

---

We use the Cherenkov light detected by icecube, the amount of pulse light, the time, and the sensor coordinates to indirectly predict the neutrino and its path.

## In the first month

### The Least Square

In the original scheme, the least square method is used. [2,3]

Because if the pulse propagated in an omni-directional manner due to a neutrino collision, the time-weighted omni-directional properties should cancel each other out, leaving only the properties of the neutrino's direction of travel.

So we replacing the charge with the mass to obtain the "center of charge" equivalent to the center of mass.

In this way, we can get two points and draw a straight line from the earliest point to the latest point.

Let $𝒗_𝒊$ be the x,y, and z coordinates of the observed value, and $𝒕𝟎_𝒊 (𝒕𝟏_𝒊)$ be the time weight, then we can simply get:

$$
v_0 = \frac{\sum(v_i \cdot charge_i \cdot t0_i)}{\sum(charge_i \cdot t0_i)}
$$
$$
v_1 = \frac{\sum(v_i \cdot charge_i \cdot t1_i)}{\sum(charge_i \cdot t1_i)}
$$

then $v_1-v_0$ is the direction vector of the neutrino, and its opposite is the direction of the source of the neutrino, that is, the value to be predicted.

Although this is a very simple method and the score is not good, this method of ideal physics model is very interesting and got me very interested in this competition in the first place.

### Graphnet and DynEdge

Then I started training a GNN from Graphnet [4], an open-source ML library for neutrino telescopes.

<div style="text-align: center;">
    <img src="../../../images/Graphnet.png" alt="Graphnet"/>
    <p style="margin-top: 1px;">Structure of Graphnet [4]</p>
</div>
The name of that GNN is DynEdge.
And one of the most important technologies is `EdgeConv`. It is a flexible al-gorithm capable of reconstructing and classifying many dif-ferent physics tasks on both a per-pulse and per-event level. It reconstructed neutrino events as point clouds and convolved over them.[5]


For every node 𝑛 𝑗 with node features 𝑥 𝑗 , the operator convolves 𝑥 𝑗 via local neighborhood of 𝑛 𝑗 as
$$
x_j = \sum_{i=1}^{N_{neighbors}} MLP(x_j - x_i)
$$

Here, $x_j$ represents the convolved node features of $n_j$, and the Multilayer Perceptron (MLP) takes as input the unconvolved node features of $n_j$ and the pairwise difference between the unconvolved node features of $n_j$ and its $i$-th neighbor. [4]

So EdgeConv receives node features and their differences from neighbors and convolves them on the graph. And its 'convolution kernel' is determined by the number of links.

This makes it particularly suited for irregular data structures like point clouds.

Most of my first month was spent understanding and retraining this vast network.

But unfortunately, I could not get a good score at that time. -- I'm depressed.

So I took a break from the competition for maybe less than a month, stopped thinking about it, and I even thought I might have failed.

## In the third month

In the last month, I switched to RNN.

And suddenly the scores started going up.

One of the very interesting ideas is to cut the zenith Angle and the azimuth Angle into 24 parts each, encode them in a base-24 way, and convert them into a classification task.

What I was doing initially was just testing various model architectures, modifying Optimizers and schedules, and so on.

Then my teammate [Enzo](https://www.kaggle.com/neomaoro) joined. It was a pleasure working with him and it was also the first time I was able to discuss so much with like-minded people during the competition.

During the discussion, we got a lot of new ideas, one of the most important is the implementation of the veto-region.

### Distinguish atmospheric muons
We distinguish between cascade events and atmospheric muon events where neutrino effects are insignificant.

<div class="image-container">
  <img src="../../../images/Muon_event.jpg" alt="Muon_event">
  <p class="image-description">(a) A muon entering the detector and leaving only a small amount of energy behind. [6]</p>
</div>
<div class="image-container">
  <img src="../../../images/Neutrino_event.jpg" alt="Neurino_event">
  <p class="image-description">(b) A cascading burst event, with most of the energy remaining in the detector [6]</p>
</div>

Every event contains neutrinos, but there are some events where neutrinos are hard to detect. For these events, we can focus on distinguishing atmospheric muons.

Regarding the separation of atmospheric muons, the main mechanism is to reconstruct their orbits and confirm whether the muons left the Earth or were shot towards it.

In principle, mesons can be excluded from the atmosphere by simple geometric considerations.

Because neutrinos can penetrate the Earth and muons cannot, muons must be shooting toward the Earth and coming from the Earth must be neutrinos.

For typical neutrino telescope installation depths, there are five to six orders of magnitude more downward moving atmospheric muons than upward moving atmospheric neutrinos.

After realizing this, we introduced the veto-zone. [7]

Then We get two ways of filtering events.

-  A. The event is filtered by Angle and rejection zone, and the event **reconstruction difficulty** is calculated according to horizontal direction, up and down, rejection zone, and weight.
-  B. According to the time window: the effective time period is calculated through the time when the pulse is strongest and the speed of light. The **rank** of a pulse is calculated according to the weight of the pulse size, and the best pulse is selected considering the pulse size.

### Training

Since method A takes into account the physical properties and method B is a simple statistical filtrate of the data, my initial assumption is that the quality of events filtered by A is better than B.

So I tried to use B for pre-training and then A for fine tuning.

The results were very poor. I was so frustrated that I thought this method might have to be abandoned.

But very coincidentally, I noticed that I was using one of the models wrong, which means I missed training one of the models.

Since the competition is close to the end, it would be waste of time to retrain, so I loaded a model that was trained on A first and trained it on B.

I thought that it would be the same as if I trained on A and then on B at first, and eventually couldn't break through. But what shocked me was that it finally broke through my best cv.

I was ecstatic, as if I had discovered a new world.

I used this method on all my models and it got me to the top3%.

##  Introspection

It definitely made me feel the fascination of science, like Prof. Francis Halzen said, '...what science is all about is not finding what you expect to find, is to finding something you didn't have any idea you were going to find, is to find the unexpected. '

But in this competition, because of my inexperience with machine learning, I still missed a lot of methods, such as stacking.

## References
[1] A. Chow, L. Heinrich, P. Eller, R. Ørsøe, and S. Dane,
IceCube - Neutrinos in Deep Ice, 2023. [Online]. Available:
https://kaggle.com/competitions/icecubeneutrinos-in-deep-ice.

[2] Mirco Hunnefield Masters Thesis, Online Reconstruction of Muon-Neutrino Events in IceCube using Deep Learning Techniques

[3] Schatto, K. Stacked searches for high-energy neutrinos from blazars with IceCube. (Mainz U., 2014).

[4] R. Abbasi, M. Ackermann, J. Adams, et al., “Graph
neural networks for low-energy event classification
& reconstruction in icecube,” Journal of Instrumentation,
vol. 17, no. 11, P11003, 2022.

[5] Y. Wang, Y. Sun, Z. Liu, S.E. Sarma, M.M. Bronstein and J.M. Solomon, Dynamic graph cnn for
learning on point clouds, ACM Trans. Graph. 38 (2019) 146.

[6] M. G. Aartsen et al., ‘Energy reconstruction methods in the IceCube neutrino telescope’, J. Inst., vol. 9, no. 03, p. P03009, Mar. 2014

[7] ICECUBE COLLABORATION, ‘Evidence for High-Energy Extraterrestrial Neutrinos at the IceCube Detector’, Science, vol. 342, no. 6161, p. 1242856, Nov. 2013

[8] Bukhari, H. et al. IceCube -- Neutrinos in Deep Ice The Top 3 Solutions from the Public Kaggle Competition. Preprint (2023).

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