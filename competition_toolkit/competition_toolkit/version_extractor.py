import tomli
import pathlib
import argparse
import re
from pyenvapi import PyenvAPI
from pyenvapi.exceptions import PyenvError

if __name__ == "__main__":
    operators = {
        "<=": lambda x: x[0]
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    file = pathlib.Path(args.config)
    if not file.exists():
        raise ValueError("Could not find config file")

    config = tomli.load(file.open("rb"))
    python_version = config["project"]["requires-python"]

    version = re.findall(r"\s*([\d.]+)", python_version, flags=re.DOTALL)[0]  # TODO can improve

    pyenv = PyenvAPI()

    possible_versions = [x for x in pyenv.available if x.startswith(version)]
    install_candidate = possible_versions.pop()

    try:
        proc = pyenv.install(version=install_candidate, verbose=True)
        proc.communicate()
    except PyenvError as e:
        pass

    print(f"{pyenv._versions_path}/{install_candidate}/bin/python")
