from git import Repo

from pathlib import Path

BASE="repos"


def clone_repository(url):

    Path(BASE).mkdir(
        exist_ok=True
    )

    name=url.split("/")[-1]

    path=f"{BASE}/{name}"

    Repo.clone_from(
        url,
        path
    )

    return path