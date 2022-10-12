import torch

def load_model( load_model_type, model_path, opts): 
    model, get_output = load_model_type(opts)
    print("this is the device I use", opts["device"])
    model.load_state_dict(torch.load(model_path, map_location=torch.device(opts["device"])))
    model.to(opts["device"])
    model = model.float()
    model.eval()
    return model, get_output