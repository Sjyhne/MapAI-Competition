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
    parser.add_argument("--models-per-ensemble", type=int, default=1, help="The maximum number of models to run simultaneously in an ensemble. Lower values use less memory, but more temp storage.")

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
        # (
        #     "https://drive.google.com/file/d/1Aqr9LnAZHMKsOZkoUp583Q3VzuTYaaUV/view?usp=share_link",
        #     "https://drive.google.com/file/d/1-id3l8kd1QwBOE6KDLb4FBadrUKypFpT/view?usp=share_link"
        # ),
        # (
        #     "https://drive.google.com/file/d/16xFkFkTgaYK5a96P1larK25749nF3G_g/view?usp=share_link",
        #     "https://drive.google.com/file/d/133TgE-Ao731rpw0pjgWn_LVoxHXBhXmt/view?usp=share_link"
        # ),
        (
            "https://drive.google.com/file/d/1DGL9xdN-E8ibjkpQAq2sfHo-mOwFJnFs/view?usp=share_link",
            "https://drive.google.com/file/d/1skM8EukONBtx2b7x1qm1XJI2AXORaEN5/view?usp=share_link"
        ),
        ( 
            "https://drive.google.com/file/d/1Hp6HAT7bTYMAt4MtQwsavNGIsUJ0z0m-/view?usp=share_link",
            "https://drive.google.com/file/d/1y1Fr1pOORX1A39ZzrDEf7VxgxlUCUPaV/view?usp=share_link"
        ),
        # (
        #     "https://drive.google.com/file/d/1C-b0TUORvbrCuDfCL6bkeTOgrw2P0eru/view?usp=share_link",
        #     "https://drive.google.com/file/d/1SwA28lxSz1MZTCLWpVFenUrFSH98kGx4/view?usp=share_link"
        # ),
        (
            "https://drive.google.com/file/d/155y9VfHUaJY5ed8Rzo4chDJ6Yx8fJ4GQ/view?usp=share_link",
            "https://drive.google.com/file/d/1NCmt2N6SToatfwSwNyZD0_H9jPhXHksM/view?usp=share_link"
        )
    ]
    
    pt_share_links2 = [
            (
            "https://drive.google.com/file/d/1iBmM3CuvKx-4CY1-7gU9jTWeuUXqvfRn/view?usp=share_link", # cp mapai resnest
            "https://drive.google.com/file/d/1yQctpXyuBgfR1gzc72yrdKRpEGD8Ojdo/view?usp=share_link" # opts
            ),
            # (
            # "https://drive.google.com/file/d/1EbSTbVADnwwuR6AYXXjK0nTtXjPeLAyU/view?usp=share_link", # mapai eb1
            # "https://drive.google.com/file/d/1FQdxYBGYkr1_NTxtFwS0XrEs2dLybiza/view?usp=share_link"
            # ),
            # (
            # "https://drive.google.com/file/d/1NtUP5QwhBglf0zSlQ0rRvIr2o4qMH8Bk/view?usp=share_link", # resnest reclassified
            # "https://drive.google.com/file/d/1ZX0H4WTdz0caX0lLNJ66D6mcoC5HsFO-/view?usp=share_link"
            # ),
            # (
            #     "https://drive.google.com/file/d/1W8PF0sKh2PVU-SzXeLJvXvheWaVF6w_a/view?usp=share_link", # eb1 lidar masks
            #     "https://drive.google.com/file/d/1whc8TQzFCaFHqBHer1vXWf1tjhPzCtZK/view?usp=share_link"
            # ),
            # (
            #     "https://drive.google.com/file/d/1RItf98I8fFOjuepgp1mPNbnXdwidbAo-/view?usp=share_link", # resnest lidar masks
            #     "https://drive.google.com/file/d/1jn_0qNke165sQCoue6PUwxkHm_bJ_32Z/view?usp=share_link"
            # ),
            # (
            #     "https://drive.google.com/file/d/1Wk83hV1rH9sLIJG8ggiMvhDfCPjRDyJb/view?usp=share_link", # eb2 reclassified
            #     "https://drive.google.com/file/d/1b0OumKfiXSMhUyqefAvbIyqLPKdnpOr-/view?usp=share_link"
            # ),
            # (
            #     "https://drive.google.com/file/d/1enih5gy9X46g_JMkfT8jYR5ssg_XNgel/view?usp=share_link",
            #     "https://drive.google.com/file/d/1p-XDUIbu-MbuISBqM43nV8JNqV_97UC1/view?usp=share_link"
            # ),
            (
                "https://drive.google.com/file/d/1JUbNYl3Z0zFMYmTs6aILj-qRKOSdT5kh/view?usp=share_link",
                "https://drive.google.com/file/d/1-QthazPuHqbJdTSen8I80r3Cug_xweKA/view?usp=share_link"
            ),
            (
                "https://drive.google.com/file/d/1WYcXT6j7o7fL4roSZeTJ3K7NY9K5xwZd/view?usp=share_link",
                "https://drive.google.com/file/d/1-kON-6jE9Yi2uADvs7ep6GFaQZpGIOv5/view?usp=share_link"
            ),
        ]
    # [0.21759022 0.28431074 0.21561935 0.0784652  0.2040145 ]
    t1_weights = [0.25070253, 0.11203359, 0.14570468, 0.25154455, 0.24001464]
    t2_weights = [0.32950561, 0.33975185, 0.33074254]

    
    args.full_submission = False
    if args.full_submission:
        pt_share_links1 = pt_share_links1[:-1]
        pt_share_links2 = pt_share_links2[:-1]

        t1_weights = t1_weights[:-1]
        t2_weights = t2_weights[:-1]


    from model_task import main as evaluate_model
    if args.task == 1:
        evaluate_model(args, pt_share_links1, t1_weights)
    elif args.task == 2:
        evaluate_model(args, pt_share_links2, t2_weights)
    else:
        args.task = 1
        evaluate_model(args, pt_share_links1, t1_weights)

        args.task = 2
        evaluate_model(args, pt_share_links2, t2_weights)

    exit(0)
