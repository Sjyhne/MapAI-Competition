import torch

def load_model( load_model_type, model_path, opts): 
    model, get_output = load_model_type(opts)
    print("this is the device I use", opts["device"])
    if opts["task"] == 2:
        new_conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.backbone.conv1 = new_conv1
    model.load_state_dict(torch.load(model_path, map_location=torch.device(opts["device"])))
    model.to(opts["device"])
    #model = model.float()
    model.eval()
    return model, get_output
