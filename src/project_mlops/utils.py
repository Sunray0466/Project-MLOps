import os

import hydra


def get_project_dir():
    print(os.getcwd())
    print(hydra.utils.get_original_cwd())
    try:
        project_dir = hydra.utils.get_original_cwd()
    except:
        if os.path.isdir("gcs"):
            project_dir = "gcs/dtumlops-bucket-group35"
        elif os.path.isdir("data"):
            project_dir = os.getcwd()
        else:
            raise Exception(f"Working directory is invalid")
    print(project_dir)
    return project_dir
