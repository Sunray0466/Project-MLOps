# Project-MLOps
**Final handin: January 24th midnight (11.59PM)**
___
## Project Description

### Goal
The overall goal of the project is to implement, utilize and make use of the tools we will be learning throughout the course. We will build an end-to-end pipeline to solve a real-world machine learning problem using opensource frameworks. The project will be mainly consisted of three parts. In the first part, we will setup an environment for the development, configure the organizational and version controlling aspects of the project. Secondly, we will train a model on a dataset and lastly, we will deploy the model to the cloud to make use of further reproducibility.

### Data
The dataset used are the <a href="https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data?select=cards.csv">playing cards from Kaggle</a>. There are 53 classes in total with 52 being the standard card set and the last being the joker. The classes have around ~150 images each and are divided into 94% training, 3% validation and 3% testing.

### Model/framework
We are planning to use “Hugging Face Transformers” or “PyTorch Image Models (TIMM)” to access and work on pre-trained image classification models. We can also try to incorporate PyTorch Lightning to our workflow to minimize boilerplate code and make it more streamlined. We first plan to use a pre-trained ResNet model with appropriate size. However, we may experiment on Vision transformer (ViT), or ConvNeXt models.

### Frontend
Link to frontend: <a href="https://frontend-474989323251.europe-west1.run.app/">https://frontend-474989323251.europe-west1.run.app/</a>

___
## #TODO
#### Checklist
* Remember to fill out the requirements.txt and requirements_dev.txt file with whatever dependencies that you are using (M2+M6)
* Remember to comply with good coding practices (pep8) while doing the project (M7)
* Do a bit of code typing and remember to document essential parts of your code (M7)
* Add command line interfaces and project commands to your code where it makes sense (M9)

#### Week 1
* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] (özkan) Fill out the data.py file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] (özkan/Sonny) Add a model to model.py and a training procedure to train.py and get that running (M6)
* [x] (Mustafa) Setup version control for your data or part of your data (M8)
* [x] (Mustafa) Construct one or multiple docker files for your code (M10)
* [x] (Mustafa) Build the docker files locally and make sure they work as intended (M10)
* [x] (Sonny) Write one or multiple configurations files for your experiments (M11)
* [x] (Sonny) Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] (özkan) Use profiling to optimize your code (M12)
* [x] (Sonny) Use logging to log important events in your code (M14)
* [x] (özkan) Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] (özkan) Consider running a hyperparameter optimization sweep (M14)
* [ ] - Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

#### Week 2
* [x] (Veroonika) Write unit tests related to the data part of your code (M16)
* [x] (Veroonika) Write unit tests related to model construction and or model training (M16)
* [x] (Veroonika) Calculate the code coverage (M16)
* [x] (Veroonika) Get some continuous integration running on the GitHub repository (M17)
* [x] (Veroonika) Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] (Veroonika) Add a linting step to your continuous integration (M17)
* [x] (Veroonika) Add pre-commit hooks to your version control setup (M18)
* [ ] - Add a continues workflow that triggers when data changes (M19)
* [x] (Veroonika) Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] (Mustafa) Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] (Mustafa) Create a trigger workflow for automatically building your docker images (M21)
* [x] (Mustafa) Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] (Sonny) Create a FastAPI application that can do inference using your model (M22)
* [x] (Sonny/özkan) Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] (Veroonika) Write API tests for your application and setup continues integration for these (M24)
* [ ] - Load test your application (M24)
* [x] (Sonny) Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] (Sonny) Create a frontend for your API (M26)

#### Week 3
* [ ] - Check how robust your model is towards data drifting (M27)
* [ ] - Deploy to the cloud a drift detection API (M27)
* [x] (Sonny) Instrument your API with a couple of system metrics (M28)
* [ ] (Mustafa) Setup cloud monitoring of your instrumented application (M28)
* [ ] (Mustafa) Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] - If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] - If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] - Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### *Extra*
* [ ] (*) Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

# How to run tests

To run the tests you must point PYTHONPATH to the project source folder
```
export PYTHONPATH=<path-to-project>/Project-MLOps/src

pytest tests/test_data.py
```
