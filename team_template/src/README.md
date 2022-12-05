# MapAI Compeition Template


## Important
Do not alter the sections in `main.py` that reads `DO NOT EDIT`. You should only include your code in the `CODE GOES HERE` section, as illustrated in the example code.
Your code should output submission to the `args.submission_path` location.

## pyproject.toml

The pyproject.toml file will be used to install the correct python version and packages
for running your project and evaluating it. It is therefore important to change it according
to the needs of your project.

There are especially a few fields that is of interest for you:

* project.name
  * Fill out the teamname with '-' as spaces (Try to find a unique name)
* project.requires-python
  * Please specify the version of python that is needed
    to run your project. E.g. `==3.8` etc.
* project.dependencies
  * Please list the project dependencies within this list.
  * You can also specify the version of each package if necessary

## Training

Train a model for task 1
> python3 train.py --task 1

Train a model for task 2
> python3 train.py --task 2

Each training run will be stored in the runs folder separated based on the task you are
training for. In the runs folder you will also find a folder with the input, prediction,
and label images - in addition to the stored model weights.

## Evaluation (NOTE!, please use the following commands, they have been altered)
To verify that your code, and to evaluate your model. You should be able to run following command:

```
mkdir -p ./submission
python3 src/main.py --task 1 --submission-path ./submission --data-type ../data/test/
python3 src/main.py --task 2 --submission-path ./submission --data-type ../data/test/
```

The commands above will output their predictions to a submission folder
which is used during evaluation on our servers

## Team info (Fill in the blank fields):

Team name: team_redpill

Team participants:  Fetullah Atas

Emails: fetullah.atas@nmbu.no

Countr(y/ies): Norway
