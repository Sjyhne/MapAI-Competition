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
    parser.add_argument("--config", type=str, default="config/main.yaml", help="Config")
    parser.add_argument("--device", type=str, default="cpu", help="Which device the inference should run on")
    parser.add_argument("--data-ratio", type=float, default=1.0, help="Percentage of the whole dataset that is used")
    parser.add_argument("--models_per_ensemble", type=int, default=1, help="The maximum number of models to run simultaneously in an ensemble. Lower values use less memory, but more temp storage.")

    args = parser.parse_args()

    #############################################################
    # CODE GOES HERE
    # Save results into: args.submission_path
    #############################################################

    pt_share_links1 = [
        (
            "https://drive.google.com/file/d/1cdFRQ12R5MziMVrE1vNKTqRL3_1Toh3x/view?usp=share_link",
            "https://drive.google.com/file/d/1jKOqHwRWNFYF67P9VFgJV9zDQ1-VJsTd/view?usp=share_link"
        ),
        (
            "https://drive.google.com/file/d/1173maCZwYTYcbbML0aGY0MCh4WXHe6p2/view?usp=sharing",
            "https://drive.google.com/file/d/1CteVOt7fatjHdobtuk3toaRMS6nYEU7s/view?usp=share_link"
        ),
        (
            "https://drive.google.com/file/d/1Aqr9LnAZHMKsOZkoUp583Q3VzuTYaaUV/view?usp=share_link",
            "https://drive.google.com/file/d/1-id3l8kd1QwBOE6KDLb4FBadrUKypFpT/view?usp=share_link"
        ),
        (
            "https://drive.google.com/file/d/16xFkFkTgaYK5a96P1larK25749nF3G_g/view?usp=share_link",
            "https://drive.google.com/file/d/133TgE-Ao731rpw0pjgWn_LVoxHXBhXmt/view?usp=share_link"
        )
    ]
    
    pt_share_links2 = [
            (
            "https://drive.google.com/file/d/1iBmM3CuvKx-4CY1-7gU9jTWeuUXqvfRn/view?usp=share_link", # cp
            "https://drive.google.com/file/d/1yQctpXyuBgfR1gzc72yrdKRpEGD8Ojdo/view?usp=share_link" # opts
            ),
            (
            "https://drive.google.com/file/d/1EbSTbVADnwwuR6AYXXjK0nTtXjPeLAyU/view?usp=share_link",
            "https://drive.google.com/file/d/1FQdxYBGYkr1_NTxtFwS0XrEs2dLybiza/view?usp=share_link"
            ),
            (
            "https://drive.google.com/file/d/1NtUP5QwhBglf0zSlQ0rRvIr2o4qMH8Bk/view?usp=share_link",
            "https://drive.google.com/file/d/1ZX0H4WTdz0caX0lLNJ66D6mcoC5HsFO-/view?usp=share_link"
            )
        ]
    
    from model_task import main as evaluate_model
    if args.task == 1:
        evaluate_model(args, pt_share_links1)
    elif args.task == 2:
        evaluate_model(args, pt_share_links2)
    else:
        args.task = 1
        evaluate_model(args, pt_share_links1)

        args.task = 2
        evaluate_model(args, pt_share_links2)

    exit(0)
