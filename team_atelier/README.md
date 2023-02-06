# Example code for team_example
This folder contain the example code that you can use as a base.
## File structure
Following is the mandatory file structure:
```
team_<NAME>
│   README.md
│   pyproject.toml
│
└───src
│   │   main.py
│   │   <other files and folders. Include program in main.py section>
```

In this particular example, we have added following custom code, that encompasses what would be a fully functional submission:
```
config/data.yml
model_task_1.py
model_task_2.py
train.py
utils.py
```
In `main.py`, you should import your code submission, e.g.,
```python
from model_task_1 import main as evaluate_model_1
from model_task_2 import main as evaluate_model_2
if args.task == 1:
    evaluate_model_1(args=args)
elif args.task == 2:
    evaluate_model_2(args=args)
else:
    evaluate_model_1(args=args)
    evaluate_model_2(args=args)
```
When you pull request to the main repository, the code is checked for errors and if everything is OK, it should pass tests and be available for merging into the main repository. When the pull-request is merged, you can consider your code delivered.
