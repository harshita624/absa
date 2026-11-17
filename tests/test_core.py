import math
import numpy as np
import torch
from absa_depth.gate import halting_distribution
from absa_depth.losses import gaussian_nll
from absa_depth.metrics import regression_metrics
from absa_depth.infer import simulate_exit, simulate_variance_exit
from absa_depth.readout import AspectReadout

def test_halting_distribution_sums_and_nonnegative():
    p=halting_distribution(torch.tensor([[.2,.4,.8],[.1,.3,.6]]))
    assert torch.all(p>=0)
    assert torch.allclose(p.sum(1),torch.ones(2))

def test_nll_matches_hand_computation():
    got=gaussian_nll(torch.tensor([[1.,2.]]),torch.zeros(1,2),torch.tensor([[3.,4.]]))
    assert torch.allclose(got,torch.tensor([4.]))

def test_readout_padding_mask_and_aspect_span():
    r=AspectReadout(4)
    h=torch.randn(1,3,4); h2=h.clone(); h2[:,2]=1e6
    mask=torch.tensor([[1,1,0]]); idx=torch.tensor([0]); asp=torch.tensor([[0,1,0]])
    a,_=r(h,mask,idx,asp); b,_=r(h2,mask,idx,asp)
    assert torch.allclose(a,b)

def test_exit_uses_first_threshold_and_depth():
    mu=torch.arange(24,dtype=torch.float32).reshape(3,4,2)
    l=torch.tensor([[.1,.7,.2,.1],[.8,.9,.1,.2],[.9,.9,.9,.9]])
    pred,d=simulate_exit(mu,l,.5)
    assert d.tolist()==[2,1,3,3]
    assert torch.equal(pred,mu[d-1,torch.arange(4)])

def test_metrics_toy_example():
    y=np.array([[1.,2.],[3.,4.]])
    m=regression_metrics(y,y)
    assert m["RMSE_V"]==0 and m["RMSE_A"]==0 and m["RMSE_VA"]==0
    assert np.isclose(m["Pearson_V"],1) and np.isclose(m["Pearson_A"],1)

def test_sentence_depth_is_max_aspect_depth():
    aspects=torch.tensor([2,4,1])
    sentence_index=torch.tensor([0,0,1])
    sentence_depth=torch.stack([aspects[sentence_index==i].max() for i in range(2)])
    assert sentence_depth.tolist()==[4,1]

def test_variance_exit_uses_first_layer_below_threshold():
    mu=torch.arange(12,dtype=torch.float32).reshape(3,2,2)
    lv=torch.log(torch.tensor([[[4.,4.],[.5,.5]],[[.2,.2],[3.,3.]],[[.1,.1],[.1,.1]]]))
    pred,depth=simulate_variance_exit(mu,lv,.75)
    assert depth.tolist()==[2,1]
    assert torch.equal(pred,mu[depth-1,torch.arange(2)])
