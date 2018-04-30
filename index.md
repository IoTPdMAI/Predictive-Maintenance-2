## Predictive Maintenance

This is a work in progress where I'll publish some brief commentary on predictive maintenance and outline my hybrid CNN-RNN model for remaining useful life modeling.

### Preliminaries

#### What is remaining useful life (RUL)?

RUL is a class of machine learning problems in which a model is trained to predict how many steps are left until a piece of machinery will fail. In my view, this is an important reframing of a very old problem, one that has normally been solved either with survival analysis or sliding window models. In the former, a Cox Proportional Hazards model (or a Weibull model, if you're feeling parametric) is used to estimate the hazard ratio for the process in question; in the latter, either an RNN or a simple logit or probit is used to estimate the instantaneous probability of failure in some predefined window.

Both of these methods are perfectly fine, but have their drawbacks. In the case of the Cox model, you're operating within the broad generalized linear model (GLM) framework, so you have to assume a linear relationship between model parameters. For the sliding window model, you need to prespecify the size of the prediction window, which can lead to some pretty arbitrary choices. 

#### WTTE-RNN as an alternative

A couple of years ago, a master's student in Sweden called Egil Martinsson came up with a brilliant new way of working on this problem. You can check out his [here](https://github.com/ragulpr/wtte-rnn), blog post [here](https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/), and thesis [here](https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf).

The long and short of it is that Egil created a new framework for modeling RUL. Rather than seeking to estimate a probability of survival, his approach uses a recurrent neural network to estimate the parameters of a Weibull distribution, which itself is used to describe the number of steps remaining until failure. This to me is incredibly cool, and I was very excited about its potential when I first encountered it - so much so that I immediately dove into some RUL problems online in an effort to validate his method and create a couple of business cases for Quantillion (the startup I currently work for).

### The PHM 2012 Challenge

#### Data

With the brief history lesson now complete, let's bring things back to the (near) present. The [PHM-2012 challenge](http://www.femto-st.fr/en/Research-departments/AS2M/Research-groups/PHM/IEEE-PHM-2012-Data-challenge.php) is the use case I decided to opt for, and your task is to use ball bearing failure data to fit a model that can predict the RUL of a bunch of other ball bearings. In principle it's pretty simple, but in reality the combination of odd time increments and an unfavorable proportion of training to test data make this pretty challenging. To cut a long story short, the approach taken was to carry out a Fourier transform on the data, normalize everything to the [-1,1] interval, and then fit an LSTM model to this data. You then end up with 1024-dimension data with a variable number of time points.

As a first step, I decided to just follow Egil's modeling approach, coding everything in PyTorch, including the loss function derived in his thesis. In PyTorch, this looks like:

``` python

class DWeibull_Loss(torch.nn.Module):
    
    def __init__(self):
        super(DWeibull_Loss, self).__init__()
    
    def forward(self, y_, a_, b_):
        hazard0 = torch.pow((y_ + 1e-35)/a_, b_)
        hazard1 = torch.pow((y_ + 1)/a_, b_)
        return -1*torch.mean(torch.log(torch.exp(hazard1-hazard0) - 1.0) - hazard1)
```
Which basically the negative log likelihood of the discrete Weibull. 

The data itself contained 8 training examples, each with between 600 and 2500 observations. However, this doesn't tell the full story, as in the actual training routine I wrote some code that randomly sampled a 100-period slice of data, which used as input to the model. The target was then the RUL of the last row in this slice of data. 

#### Model 1

![GitHub Logo](/exp2.png)
