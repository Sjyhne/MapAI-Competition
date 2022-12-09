import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #############################################################
    # Mandatory arguments. DO NOT EDIT
    #############################################################
    parser.add_argument("--submission-path", required=True)
    parser.add_argument("--data-type", required=True, default="validation", help="validation or test")
    parser.add_argument("--task", type=int, default=3, help="Which task you are submitting for")

    #############################################################
    # CUSTOM ARGUMENTS GOES HERE
    #############################################################
    parser.add_argument("--config", type=str, default="config/data.yaml", help="Config")
    parser.add_argument("--device", type=str, default="cpu", help="Which device the inference should run on")
    parser.add_argument("--data-ratio", type=float, default=1.0, help="Percentage of the whole dataset that is used")

    args = parser.parse_args()

    #############################################################
    # CODE GOES HERE
    # Save results into: args.submission_path
    #############################################################
    from model_task_1 import main as evaluate_model_1
    from model_task_2 import main as evaluate_model_2
    if args.task == 1:
        evaluate_model_1(args=args)
    elif args.task == 2:
        evaluate_model_2(args=args)
    else:
        evaluate_model_1(args=args)
        evaluate_model_2(args=args)

    exit(0)
