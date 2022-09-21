import argparse
import json

import jinja2
from jinja2 import FileSystemLoader
import markdown
import pathlib
from pydantic import BaseModel
import tomli
import os
import urllib.request
import numpy as np

base_dir = pathlib.Path(__file__).parent

class Participant(BaseModel):
    name: str = None
    description: str = None
    python_version: str = None
    error: str = None
    iou1: float = None
    biou1: float = None
    iou2: float = None
    biou2: float = None
    score: float = None


def get_competition_root(prefix: str):
    for p in base_dir.parents:
        if not any([x for x in p.iterdir() if f"{prefix}_" in x.name]):
            continue
        return p


def property_exists(d: dict, key: str):
    if key not in d:
        raise ValueError(f"Key: '{key}' does not exist in dictionary.")


def get_participants(prefix: str):
    competition_root: pathlib.Path = get_competition_root(prefix=prefix)
    participant_paths: list[pathlib.Path] = list(competition_root.glob(f"*{prefix}_*"))
    participants = []
    for participant_path in participant_paths:
        try:
            participant_data = parse_participant(participant_path)
        except ValueError as e:
            participant_data = dict(error=str(e), name=participant_path.name)

        participants.append(Participant.parse_obj(participant_data))
    return participants


def parse_participant(participant_path: pathlib.Path):
    project_file = participant_path.joinpath("pyproject.toml")

    if not project_file.exists():
        raise ValueError("Could not find pyproject.toml")

    with project_file.open("rb") as f:
        participant_config = tomli.load(f)

    property_exists(participant_config, "project")
    property_exists(participant_config["project"], "name")
    property_exists(participant_config["project"], "description")
    property_exists(participant_config["project"], "requires-python")

    try:
        task1 = json.load(open(os.path.join("evaluation_results", str(participant_path).split("/")[-1] + "_task_1.json"), "r"))
        task1iou = task1["iou"]
        task1biou = task1["biou"]
    except Exception as e:
        task1iou = 0.0
        task1biou = 0.0

    try:
        task2 = json.load(open(os.path.join("evaluation_results", str(participant_path).split("/")[-1] + "_task_2.json"), "r"))
        task2iou = task2["iou"]
        task2biou = task2["biou"]
    except Exception as e:
        task2iou = 0.0
        task2biou = 0.0

    score = np.round((task1iou + task1biou + task2iou + task2biou)/4, 5)

    return dict(
        name=participant_config["project"]["name"],
        description=participant_config["project"]["description"],
        python_version=participant_config["project"]["requires-python"],
        iou1=task1iou,
        biou1=task1biou,
        iou2=task2iou,
        biou2=task2biou,
        score=score
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="build", type=str)
    parser.add_argument("--competition-title", default="competition_title", type=str)
    parser.add_argument("--prefix", default="team", type=str)
    args = parser.parse_args()

    participants = get_participants(args.prefix)

    # Sort participants based on score
    participants.sort(key=lambda x: x.score, reverse=True)

    #
    readme_data = urllib.request.urlopen("https://raw.githubusercontent.com/Sjyhne/MapAI-Competition/master/README.md").read().decode()

    environment = jinja2.Environment(loader=FileSystemLoader(base_dir.joinpath("templates/")))

    template = environment.get_template("index.html")
    render = template.render(
        participants=participants,
        competition_title=args.competition_title,
        readme_data=markdown.markdown(readme_data)
    )

    outfile = pathlib.Path(args.out)
    outfile.mkdir(exist_ok=True)
    outfile.joinpath(template.name).write_text(render)
