from __future__ import absolute_import
from collections import OrderedDict


from ..utils import to_torch

def extract_cnn_feature(model, inputs, modules=None):
    model.eval()
    # with torch.no_grad():
    inputs = to_torch(inputs).cuda()
    if modules is None:
        # outputs, original_feas, att_scores = model(inputs)
        outputs, prob, original_feas, att_scores = model(inputs) # select add
        # print("outputs: ", outputs.shape)
        # print("original_feas ", original_feas.shape)
        # print("att_scores: ", att_scores.shape)
        outputs = outputs.data.cpu()
        original_feas = original_feas.data.cpu()
        att_scores = att_scores.data.cpu()
        return outputs, original_feas, att_scores
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())
