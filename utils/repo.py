import os

from git import Repo


def _get_current_repo():
    repo = Repo(os.getcwd())
    return repo


def save_repo_archive(filename):
    repo = _get_current_repo()

    with open(filename, 'wb') as fp:
        repo.archive(fp)
