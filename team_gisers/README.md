# Code for team_GISers
This folder contain the submission code.
## File structure
Following is the submitted file structure (and some introductions for the specific files):
```
team_GISers
│   README.md
│   pyproject.toml    
│
└───src
│   │   configs/
│   │   models/
│   │   main.py
│   │   model_task_1.py     -- without auto device switching between cpu and gpu, when loading weights file
│   │   model_task_1_v2.py  -- WITH auto device switching between cpu and gpu, when loading weights file
│   │   model_task_2.py     -- without auto device switching between cpu and gpu, when loading weights file
│   │   model_task_2_v2.py  -- WITH auto device switching between cpu and gpu, when loading weights file
│   │   utils.py
│   │   README.md
```

---------------------------------------

##### local test environment is:
ubuntu 20.04
python==3.8.10
torch==1.11.0+cu113 or 1.13
gdown==latest