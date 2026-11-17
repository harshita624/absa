import torch
from transformers import BertConfig, BertModel
from absa_depth.model import AdaptiveDepthModel
from absa_depth.infer import predict_batched, predict_true_early

class TinyTokenizer:
    def __call__(self,text,**kwargs):
        return {"input_ids":torch.tensor([[101,200,201,102]]),
                "attention_mask":torch.tensor([[1,1,1,1]]),
                "token_type_ids":torch.zeros((1,4),dtype=torch.long),
                "offset_mapping":torch.tensor([[[0,0],[0,4],[5,9],[0,0]]])}

def test_batched_and_true_incremental_exit_match():
    torch.manual_seed(7)
    config=BertConfig(vocab_size=300,hidden_size=16,num_hidden_layers=3,num_attention_heads=4,
                      intermediate_size=32,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
    model=AdaptiveDepthModel.__new__(AdaptiveDepthModel)
    torch.nn.Module.__init__(model)
    model.encoder=BertModel(config,add_pooling_layer=False)
    model.hidden_size=16; model.layers=3; model.gate_detach_inputs=True
    from torch import nn
    from absa_depth.readout import AspectReadout
    from absa_depth.gate import HaltingGate
    model.readout=nn.ModuleList([AspectReadout(16) for _ in range(3)])
    model.heads=nn.ModuleList([nn.Sequential(nn.Linear(32,16),nn.GELU(),nn.Linear(16,4)) for _ in range(3)])
    model.gate=HaltingGate(32,3)
    model.eval()
    batch={"input_ids":torch.tensor([[101,200,201,102]]),"attention_mask":torch.ones((1,4),dtype=torch.long),
           "token_type_ids":torch.zeros((1,4),dtype=torch.long),
           "offset_mapping":torch.tensor([[[0,0],[0,4],[5,9],[0,0]]]),
           "sentence_index":torch.tensor([0,0]),"char_starts":torch.tensor([5,0]),"char_ends":torch.tensor([9,4])}
    fast,d1=predict_batched(model,batch,.45)
    slow,d2=predict_true_early(model,TinyTokenizer(),"good food",["food","good"],.45,torch.device("cpu"))
    assert torch.equal(d1,d2)
    assert torch.allclose(fast,slow,atol=1e-6)
