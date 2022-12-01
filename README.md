# NORA MapAI: Precision in Building Segmentation

<a href="https://journals.uio.no/NMI/article/view/9849"><img src="https://img.shields.io/badge/Competition%20Paper-MapAI-brightgreen" ></a>
<a href="https://sjyhne.github.io/MapAI-Competition/"><img src="https://img.shields.io/badge/Competition-Results-brightgreen" ></a>
<a href="https://huggingface.co/datasets/sjyhne/mapai_training_data"><img src="https://img.shields.io/badge/MapAI-Dataset-brightgreen" ></a>


## Important dates (Final! Deadline Extension)

* Participant's submission of results: ~~25th of November~~ -> ~~2nd of December~~ -> Extended to **_5th of December_**.
* Feedback on the evaluation results: ~~5th of December~~ -> ~~9th of December~~ -> Extended to **_12th of December_**.
* Deadline for the 2-pager description paper: ~~15th of December~~ -> Extended to **_22nd of December_**.


## Update!

I was notified about a flaw in the evaluation functionality currently implemented, where the evaluation disregards
the IoU of the background predictions. The issue is now fixed. Additionally I introduce a method for checking whether
the current version of your fork is up-to-date with the newest version available on Github. Please see [Check Fork Version](https://github.com/Sjyhne/MapAI-Competition#check-fork-version)
in the readme.

This is the official repository for the MapAI competition arranged by 
the Norwegian Mapping Authority, Centre for Artificial Intelligence Research at University of Agder (CAIR),
Norwegian Artificial Intelligence Research Consortium (NORA), AI:Hub, Norkart, and the Danish Agency for 
Data Supply and Infrastructure.

For this competition we are testing a new competition-framework/template for arranging 
competitions using Github and Huggingface. This is the first competition we are arranging with
the framework, and therefore appreciate your feedback and will be available for questions.
If there is a question or something is wrong, please raise a issue regarding the question.

The competition is sponsored by AI:Hub and The Norwegian Mapping Authority

<div style="text-align:center">
<img src="https://github.com/Sjyhne/MapAI-Competition/blob/master/figures/aihub_logo.png?raw=true" alt="aihub logo" width="200" style="padding: 20px"/>
<img src="https://github.com/Sjyhne/MapAI-Competition/blob/master/figures/kartverket_logo.png?raw=true" alt="kartverket logo" width="200" style="padding: 20px"/>
</div>


## Competition Information

The competition aims at building segmentation using aerial images and lidar data. The 
competition consists of two different tasks:

1. Building segmentation only using aerial images.
2. Building segmentation using lidar data (it is allowed to combine the lidar with aerial images).
### Task Rules

You are allowed to use additional training data for both tasks. The datasets used for training must be open and accessible to everyone and follow the specific rules for each task.

**Task 1:**
- You are allowed to use pretrained models trained on RGB images which are open and accessible to everyone. 
	- E.g. ImageNet pretrained backbones/networks.
- You are allowed to use other open datasets for training the models, however, they must be datasets with RGB-images. Therefore, you cannot use other types of data for training the models for this specific task.

**Task 2:**
- You are allowed to use pretrained models trained on RGB images and/or LiDAR data which are open and accessible to everyone.
	- E.g. ImageNet pretrained backbones/networks.
- You are allowed to use other open datasets for training the models, however, they must be datasets with RGB-images and/or LiDAR-data. 

To be eligible for prizes both the source-code and a 2-page paper must be submitted, allowing us to verify the submission according to the rules.

### Dataset

The training and validation data is an open building segmentation dataset from Denmark. While the
test data comes from various locations in Norway which is currently not released yet. All
data will be released after the competition in a collected huggingface dataset.

It is important to note that the training and evaluation data comes from real-world data. As a
result there will be flaws in the dataset, such as missing masks or masks that does not correspond
to a building in an image. The images come from orthophotos generated using a DTM, and therefore, the building-masks are
slightly skewed compared to the ground truth masks.

The dataset is hosted on Huggingface and can be found [here](https://huggingface.co/datasets/sjyhne/mapai_training_data).
Downloading the dataset is done using the huggingface datasets package.

First you need to install the competition_toolkit package with pip (command expects your current directory is MapAI-Compeition):

> pip3 install competition_toolkit/

The you can download the dataset with the following python code, be sure to run the function from _within_ the src folder
in your team folder:


```python
from competition_toolkit.dataloader import download_dataset
data_type = "train" # or "validation"
task = 1 # or 2, but does not matter for train and validation data types
# paths is a list of dicts, where each dict is a data sample, 
# and each dict have "image", "lidar", and "mask" keys
# with the path to the corresponding file
paths = download_dataset(data_type, task)

data_sample = paths[0]

"""
data_sample = {
    "image": <path_to_img>,
    "lidar": <path_to_lidar>,
    "mask": <path_to_mask>
}
"""
```

where data_type can either be "train" or "validation". The function downloads a .parquet file from huggingface
and extracts images, masks, and lidar data to a data/ folder in the base of the repository. 

NB: If the function returns paths without downloading them, and you do not have a data/ folder in the MapAI-Competition
folder, then you have to remove the cache for huggingface datasets. In Ubuntu, you will find this cache in `~/.cache/huggingface`

### Motivation

Acquiring accurate segmentation masks of buildings is challenging since the training data 
derives from real-world photographs. As a result, the data often have varying quality, 
large class imbalance, and contains noise in different forms. The segmentation masks 
are affected by optical issues such as shadows, reflections, and perspectives. Additionally, 
trees, powerlines, or even other buildings may obstruct the visibility. 
Furthermore, small buildings have proved to be more difficult to segment than larger ones as 
they are harder to detect, more prone to being obstructed, and often confused with other classes. 
Lastly, different buildings are found in several diverse areas, ranging 
from rural to urban locations. The diversity poses a vital requirement for the model to 
generalize to the various combinations. These hardships motivate the competition and our 
evaluation method.

### Registration

Forking the base repository is part of the registration process. However, we would also like you to register your information
in a Google forms. The main reason comes from the ability to reach out to the participant with important information. 
As mentioned earlier, the competition framework is piloting with this competition, and therefore it will be more
prone to change compared to more established competition frameworks. Please fill out the form below.

[Google form](https://docs.google.com/forms/d/1sJ9R2j32dc7TUDDkVCpLsaIVBcxRcOLiwmqeq-15O5E/edit)

### Prizes

The prizes will be 1200 euros for first place, 500 euros for second place, and 300 Euros for third place.

1. 1200 Euro
2. 500 Euro
3. 300 Euro

Results will be presented at https://sjyhne.github.io/MapAI-Competition/ after evaluation.

### Check Fork Version

To verify that you're currently using the latest fork version, you can run the `check_latest_version.py` file
and it will print whether or not you have to take action.

If it says that you need to update, then you can update your repo with the following steps:

1. ```git remote add upstream git@github.com:Sjyhne/MapAI-Competition.git```
2. ```git fetch upstream```
3. ```git rebase upstream/master```
4. ```git push origin master```
   * It might be necessary to force push it to your own repository: ```git push -f origin master```

## Instructions

The competition will be arranged on Github. The steps for participation is as following:

### Steps

#### Step 1 - Fork

Fork the [MapAI-Competition](https://github.com/Sjyhne/MapAI-Competition) repository in Github.
Forking creates a clone of the base repo on your own user and allows for easier pull requests
and so on.

#### Step 2 - Clone with -o parameter

Clone your fork down to your computer with the following command:

`git clone git@github.com:<your_username>/MapAI-Competition.git -o submission`

The _-o_ parameter sets the origin name for this repostory to be "_submission_" and not the
default which is "_origin_".

#### Step 3 - Create a new private (or public) repository

Create a new private repository on your own github. The reason we need this is because it is
not possible to set the visibility of a fork to private. Therefore, to keep your development progress
private, we have to add another remote repository for the MapAI-Competition fork.

To do this, you have to change directories into the cloned fork. E.g. `cd MapAI-Competition`.

#### Step 4 - Add private remote repository to fork

Then, we can keep developing in the cloned fork and push the changes to the private repository.
To be able to do this, we have to add remote origin by running the following command:

`git remote add origin <private_repository>`

E.g.

`git remote add origin git@github.com:Sjyhne/my_private_repository.git`

This will enable you to push your changes to the private repository and not the public fork
by just pushing as usual to origin master. Because we have not specified the origin for the remote 
it will default to _origin_.

`git push origin <branch>`

#### Step 5 - Create your own team-folder

It is important to follow the structure of the team_template in the repository. The easiest way to
keep this structure is by just creating a copy of the team_template folder and name it according
to your team name. The folder you create must follow the correct naming structure, which is 
`team_<team_name>` (Please make it kinda unique, e.g. two first letters of each teammate). 
You can copy the team_template and name it with the following command:

`cp -r team_template ./team_<team_name>`

For the entirety of the competition, you will only change and develop inside this folder. Nothing
outside the team-folder should be changed or altered. You can find more information about
the folder structure and contents in the section about _folder structure_.

The template is already ready with code that can run, train, and evaluate - this is just template
code and you are allowed to change everything related to the training of the models. 
To use the template code, some functionality relies on the competition_toolkit package which can
be installed from the MapAI-Competition folder with the following command:
> pip3 install competition_toolkit/

When it comes the evaluation file, it is more restricted, as they are used to automatically 
evaluate the models.

**Please** fill out the all of the details in the form at the bottom of the README.md inside your src folder.

#### Step 6 - Delivery

When the deadline is due, there are a few steps that will have to be taken to get ready for
submission.

NB: The models will be evaluated against images with **500x500** resolution. Therefore, the predictions of the models **must** be the same size. The model can produce 500x500 predictions directly, or you can resize the  predictions to the correct size.

##### Step 6.1 - Push your changes to the fork

Push all of your changes to the fork - this will make your code and models visible in the fork.
This is done by running the following command:

`git push submission master`

As we set the origin for the fork to _submission_ in the start.

##### 6.2 - Create a pull request to the base repo

The next step is to create a pull request against the base repository. This will initiate a 
workflow that runs and evaluates the model on a validation dataset. This workflow will have to
pass in order to deliver the submission.

When the deadline is finished, we will evaluate all of your code on the hidden test-dataset and publish the results
on a github page.

#### Step 7 - 2 Page Paper

All participants are asked to submit a 2 page paper (double column, plus 1 additional page
for references) describing their method and results. The 2-pager must include a description 
of the datasets that have been used to train the models. The submitted papers will be 
reviewed single blind and will be published. Outstanding submissions will be invited 
to submit a full length paper to a special issue about the competition in the Nordic 
Machine Intelligence Journal.

### Uploading and downloading models from drive.google.com

_This is optional, and we're satisfied as long as the models are downloaded and loaded through the script_

As the .pt files for models can be rather large, we want you to upload the model files to
your own google drive and download them from there during evaluation.

1. Train a model and get the .pt file
2. Upload the modelfile to a google drive folder
3. Enable sharing with link (Which allows us to use gdown to download the modelfile)
4. Get the link
   * Most likely looking like: "https://drive.google.com/file/d/1wFHRUDe29a82fof1LNwFsFpOvX0StWL5/view?usp=sharing"
   * Then just put the sharing link into the _pt_share_link_ variable in both evaluate_task_*.py files which will
     get the id and put it into the correct format
5. Test the modelfiles and check that it is correct and loads correctly
6. During submission, ensure this works correctly by following the "Checklist before
    submission" section

### Checklist before submission

Due to some models being larger than 8GB, and therefore cannot be run on the default Github runner,
you can either run the evaluation pipeline, or you can follow the steps specified below for verifying
that it can be run on our local pipeline.

* Verify that you are able to run the evaluation pipeline
  1. To do this you can go to your own repository.
     * Then go to the actions tab
     * Press the "Evaluation Pipeline"
     * Then press the "run workflow" for the branch you are testing (most likely master)
     * Ensure the entire workflow runs without any issues (environment issues are common)
       * If the environment issues are an issue, then you have to edit the pyproject.toml in
         the base of your team folder

* For local verification of being able to run in the evaluation pipeline
   * Extract the specified python version from your .toml file
     * `python3 competition_toolkit/competition_toolkit/version_extractor.py --config <team_folder>/pyproject.toml`
   * Then create a virtual environment with the python version specified
     * `<python_version> -m venv env`
   * Source the virtual environment so you can install pip dependencies
     * source env/bin/activate
   * Pip install the packages specified in the .toml file which is necessary to run your code
     * `pip3 install <team_folder>`
   * Pip install the competition toolkit
     * `pip3 install competition_toolkit/`
   * Create the submission folder
     * `submission_path="/tmp/MapAI-<team_folder>-submission`
     * `rm -rf $submission_path`
     * `mkdir -p $submission_path`
   * Change directories into the src folder of your team folder
     * `cd <team_folder>/src`
   * Now see that task 1 are running fine (If you are submitting for task 1)
     * `<python_version> main.py --data-type val --submission-path $submission_path --task 1`
     * `<python_version> ../../competition_toolkit/competition_toolkit/evaluation.py --task 1 --submission-path $submission_path --team <team_folder> --data-type val`
   * Now see that task 2 are running fine (If you are submitting for task 2)
     * `<python_version> main.py --data-type val --submission-path $submission_path --task 2`
     * `<python_version> ../../competition_toolkit/competition_toolkit/evaluation.py --task 2 --submission-path $submission_path --team <team_folder> --data-type val`
* It is important that when ensuring that task 1 and task 2 are running fine, the model is loaded from
    a cloud provider or similar, an example is using Google drive as specified above.

### Bibtex Citation

```
@article{Jyhne2022,
   author = {Sander Jyhne and Morten Goodwin and Per-Arne Andersen and Ivar Oveland and Alexander Salveson Nossum and Karianne Ormseth and Mathilde Ørstavik and Andrew C Flatman},
   doi = {10.5617/NMI.9849},
   issn = {2703-9196},
   issue = {3},
   journal = {Nordic Machine Intelligence},
   keywords = {Aerial Images,Deep Learning,Image segmentation,machine learning,remote sensing,semantic segmentation},
   month = {9},
   pages = {1-3},
   title = {MapAI: Precision in Building Segmentation},
   volume = {2},
   url = {https://journals.uio.no/NMI/article/view/9849},
   year = {2022},
}
```