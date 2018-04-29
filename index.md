## Predictive Maintenance

This is a work in progress where I'll publish some brief commentary on predictive maintenance and outline my hybrid CNN-RNN model for remaining useful life modeling.

### Preliminaries

#### What is remaining useful life (RUL)?

RUL is a class of machine learning problems in which a model is trained to predict how many steps are left until a piece of machinery will fail. In my view, this is an important reframing of a very old problem, one that has normally been solved either with survival analysis or sliding window models. In the former, a Cox Proportional Hazards model (or a Weibull model, if you're feeling parametric) is used to estimate the hazard ratio for the process in question; in the latter, either an RNN or a simple logit or probit is used to estimate the instantaneous probability of failure in some predefined window.

Both of these methods are perfectly fine, but have their drawbacks. In the case of the Cox model, you're operating within the broad generalized linear model (GLM) framework, so you have to assume a linear relationship between model parameters. For the sliding window model, you need to prespecify the size of the prediction window, which can lead to some pretty arbitrary choices. 

#### WTTE-RNN as an alternative

A couple of years ago, a master's student in Sweden called Egil Martinsson came up with a brilliant new way of working on this problem. You can check out his [here](https://github.com/ragulpr/wtte-rnn), blog post [here](https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/), and thesis [here](https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf).
