import argparse
from yaml import load, Loader

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_opts():
    parser = argparse.ArgumentParser("Training a segmentation model")

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate used during training")
    parser.add_argument("--config", type=str, default="team_morty/src/config/data.yaml", help="Configuration file to be used")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--data_ratio", type=float, default=1.0,
                        help="Percentage of the whole dataset that is used")

    args = parser.parse_args()

    # Import config
    opts = load(open(args.config, "r"), Loader)
    # Combine args and opts in single dict
    try:
        opts = opts | vars(args)
    except Exception as e:
        opts = {**opts, **vars(args)}
    return opts