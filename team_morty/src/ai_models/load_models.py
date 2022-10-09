import torch

def load_model( load_model_type, model_path, opts): 
    model, get_output = load_model_type(opts)
    model.load_state_dict(torch.load(model_path))
    model.to("cuda")
    model = model.float()
    model.eval()
    return model, get_output