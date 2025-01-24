# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [ ] Make sure that all team members have write access to the GitHub repository (M5)
* [ ] Create a dedicated environment for you project to keep track of your packages (M2)
* [ ] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [ ] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [ ] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [ ] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [ ] Do a bit of code typing and remember to document essential parts of your code (M7)
* [ ] Setup version control for your data or part of your data (M8)
* [ ] Add command line interfaces and project commands to your code where it makes sense (M9)
* [ ] Construct one or multiple docker files for your code (M10)
* [ ] Build the docker files locally and make sure they work as intended (M10)
* [ ] Write one or multiple configurations files for your experiments (M11)
* [ ] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [ ] Use logging to log important events in your code (M14)
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [ ] Write unit tests related to the data part of your code (M16)
* [ ] Write unit tests related to model construction and or model training (M16)
* [ ] Calculate the code coverage (M16)
* [ ] Get some continuous integration running on the GitHub repository (M17)
* [ ] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [ ] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [ ] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>Answer:

G35

### Question 2
> **Enter the study number for each member in the group**
> Answer:

s250069, s204133, s242540

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Answer:
>
>Among the many different tools we used, PyTorch Image Models (TIMM) and Kaggle were the ones that were not directly covered by the course material. We used Kaggle to browse through many datasets, and to find a dataset that is both intriguing to work on and managable in size. We wanted to work on an image classification project and we chose to work with the playing cards dateset, wihich can be found by [clicking here](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data). The dataset has 7624 training images, 265 test images and 265 validation images from 53 classes (52 normal playing cards + 1 joker).
>
>We used TIMM framework to get a pretrained ResNet18 model in PyTorch, and fine-tune it. We were able to freeze its initial layers and run training on a few of the layers from the end. This enabled us to get a good performing model with relatively short training times. This enabled us to more easily focus to the operations related parts of this MLOps project.


## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
> Answer:

We managed dependencies in our project using a combination of a requirements.txt file and a pyproject.toml file. The requirements.txt lists all the exact package versions used, ensuring a reproducible environment. Meanwhile, the pyproject.toml file defines broader dependency specifications, including build system requirements. To replicate the environment, a new team member can create a virtual environment, then install the dependencies using pip install -r requirements.txt for precise versions or pip install . if they want to build the project as defined in the pyproject.toml. This ensures consistency across development, testing, and production environments.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
> Answer:

With cookiecutter template, we copied the course template and have since then filled out most of the existing folders except the notebook-folder. All folders were kept and none were deleted, but new additions have also been made. The *.dvc* for data control, *.ruff* for formatting, *deployment* for our back- and frontend, and *outputs* and *wandb* for hydra and weights and biases, which keeps track of our hyperparameters.

The cookiecutter overall made it easy for us to set up a general structure for our work without hassle and it was easy to figure out where we should go to to add new files and folders without later moving them again.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
> Answer:

We used Ruff for linting and formatting in our Python project. Running Ruff checks and formatting during development helps maintain good code quality by enforcing consistent styling and catching potential issues early, ensuring the code remains readable, maintainable, and aligned with best practices. Integrating Ruff into a pre-commit hook made it easy to spot and fix mistakes before committing, which prevented problematic changes from reaching the shared codebase and minimized the time spent on code reviews. Additionally, including Ruff in our GitHub Actions or continuous integration (CI) pipeline provided an extra layer of validation by ensuring that all code merged into production adhered to the established standards and was free of basic errors. This end-to-end integration helped us avoid regressions, maintain clean code, and improve overall efficiency in the development process.

In larger projects, these concepts are essential because inconsistencies in coding style can make the code harder to read, understand, and maintain. Proper linting ensures that the code follows a shared standard, making collaboration smoother and reducing the chances of bugs slipping through. For example, Ruff's speed and ability to handle large codebases efficiently enabled us to enforce rules quickly, saving time and keeping our project clean as it scaled.


## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
> Answer:

In total, we implemented 13 tests across three test files: 'test_api.py', 'test_data.py', and 'test_model.py', ensuring functionality, reliability, and consistency in our codebase. We used pytest for testing, as it provides a robust framework for writing and running tests efficiently. Additionally, we integrated these tests into GitHub Actions to automate testing for every pull request, ensuring continuous integration and maintaining high code quality throughout the development process.
test_api.py: This file includes three tests to validate the FastAPI endpoints for our image classification API. We test the root endpoint to confirm it responds correctly, ensure invalid file uploads return the expected error status, and verify the proper functionality of the '/classify/' endpoint using an actual image. These tests ensure that the API behaves as intended and handles edge cases gracefully.
test_data.py: This file focuses on testing our Kaggle playing cards dataset. It validates dataset sizes, checks image shapes, ensures all 53 card classes are represented, and verifies target values remain within the expected range. These tests guarantee the integrity and usability of the dataset for training and evaluation.
test_model.py: This file includes seven tests to verify the initialization, forward pass, and architecture of our models. It ensures that models handle inputs correctly, contain the expected components (e.g., dropout layers), and use pretrained weights appropriately. It also validates parameter counts and tests for handling invalid input shapes.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
> Answer:

The total code coverage for the tests is approximately 56%, based on the following: test_model.py covers 100%, test_data.py covers 40%, and test_api.py covers 30%. This indicates that while some parts of the code are thoroughly tested, significant portions remain untested.

Even with 100% code coverage, the code cannot be considered error-free. Code coverage measures how much of the code is executed during tests but does not guarantee that all edge cases, integration issues, or logical errors are addressed. High-quality tests that validate correctness and handle various scenarios are essential for ensuring reliability, not just achieving high coverage.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Answer:
> 
>In our project, we tried both PRs and branches as much as we could. We tried to split the workload of the project, and work on respective branches. We tried to utilize the PRs while merging the feature branches to the main branch. It was sometimes okay to push our individual changes directly to main. However, in bigger projects benefits of branches and PRs surpass their additional overhead work.
>
>Seperate branches to make development on seperate features enables us to make changes on our isolated instant of the program. This enables us to work on different aspects of the project without interfering with each other's work. This enables us to alleviate code conflicts in bigger projects, and keeps our version history clear and tidy. For example, If we choose to squash the commits in a feature branch while merging it to main branch, we keep our main branch history clear and easy-to-read.  
>
>Using PRs enables relevant people to engage in code review, and helps them to collaborate. It also makes it easier to inspect the changes we are about to introduce to the target branch. This can be useful to prevent potential catastrophes in big and complex projects.


### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
> Answer:

We used DVC in our project to manage data, but it didn’t significantly improve our workflow since our dataset was static and downloaded from Kaggle. DVC's main benefits, like tracking evolving datasets or maintaining different versions, weren’t applicable in our case.

However, DVC would be highly beneficial in scenarios involving dynamic datasets, such as when a team iteratively collects and preprocesses data. For example, in a machine learning project with continuously updated data (e.g. IoT sensor data), DVC enables efficient tracking of data changes. It creates a synchronization across team members, and reproducibility by linking data to specific model versions, ensuring consistency in experiments and deployments.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
> Answer:

Our continuous integration (CI) setup leverages GitHub Actions to ensure code quality and maintain stability during development. We have structured our CI into a single workflow file, tests.yaml, which focuses on automating the testing process.
This setup includes:
Unit Testing: The workflow is triggered on every push or pull_request to the main branch. It runs unit tests located in the tests folder (test_api.py, test_data.py, test_model.py) using pytest, and coverage is calculated with coverage.

Environment Support: Our CI tests on macos-latest with Python 3.11. While the current setup is limited to one operating system and Python version, it can easily be extended to test across multiple platforms (e.g., Windows, Linux) or Python versions by modifying the matrix strategy.

Dependency Management and Caching: Dependencies are installed using pip, with caching enabled to speed up subsequent builds. The pip cache uses setup.py to track dependency changes and ensure up-to-date installations without redundant downloads.

We use the Python package Coverage to generate a code coverage report locally after running tests. This report is then uploaded to Codecov, which works seamlessly with GitHub Actions to integrate coverage insights into our pull requests.

External Resources: The workflow integrates with Kaggle by using environment variables to download required datasets before running tests.
This streamlined setup ensures consistent, reliable testing and reduces manual effort in maintaining code quality. It can be further extended to include additional features, such as linting (e.g., Ruff), multiple OS testing, or integration tests.

Link: https://github.com/Sunray0466/Project-MLOps/actions/runs/12956631076/job/36143337658


## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
> Answer:

At first we had "typer" to send arguments to our script which works fine. We then introduced Hydra which did not work with typer and we ended up with discarding typer. Hydra gave us a lot of issues as it apparently changes the working directory which we found out *much later*. After fixing these issues Hydra would then accept our arguments as well as log our parameters, arguments, performance and so on which works great. And then we tried using WANDB... We then had a lot of troubleshooting until we finally found out how we can combine both of them to get the best of both world.

In order to execute our training we could either run:<br>
`python .\src\project_mlops\train.py lr=0.02 batch_size=10` or `wandb sweep .\configs\sweep.yaml` followed by the command sweep gives.


### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:
> We used Hydra's config file to feed hyperparameters to our 'train.py' training file. This way we do not have to manually give command-line arguments every time we run the training file. We used 'logging' package to write some of our console prints to seperate files each time we run training. This way we can check the results of past runs and have look at the hyperparameters we used then. This is also amplified with logging some training metrics to WandB and saving a copy of the trained model there. We also used torch.manual_seed() function of PyTorch to lock in the randomness between different training runs. This makes it easier to see the effects of other aspects and parameters on training performance 
>
>Another important aspect of reproducibility is using docker images together with virtual environments and requirement.txt files. This way we can replicate our local runs in cloud deployments, without breaking a sweat about hardware and operating system differences.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Answer:

![my_image](figures/pb-133-1.png)

In the image above we see important training metrics such as train loss, train accuracy, validation loss and validatilon accuracy. These enable us to track the model's performance throughout the training. Using the validation set is especially important, because we need to check the model performance with data that is not used in parameter optimization. Monitoring validation and training together gives us an idea on whether it is overfitting, and whether the gradients are behaving well throughout training. 

![my_image](figures/pb-133-2.png)

In the image above, we see some sample images used in training. It enables us to see that we are not using completely irrelevent and corrupted images. 

We also see the graphs of some external metrics throughout training, such as GPU utilization and power consumption. In large scale projects power consumption might be an important aspect of the running costs ıf the project. CPU utilization can give us an idea about the model complexity and batch sizes we can work with.

![my_image](figures/sweep.png)

In the image above we see some metrics coming from a hyperparameter sweep. We can see how it behaves with different sets of hyperparameters during training. Especially the plot at the bottom can be very useful. It enables us to see the final validation accuracies of different runs with different hyperparameters, in a single plot. 

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
> Answer:

Docker played a crucial role in our project for both training and deployment of the backend and frontend components. For training, we created a Docker image that contained all the dependencies and configurations required to run our machine learning training script. We set up an entry point in the Dockerfile to ensure seamless execution within Vertex AI, where the image was used to execute training jobs.

For deployment, we used Docker images to containerize both the backend (FastAPI) and frontend (Streamlit) implementations. These images were built and pushed to Artifact Registry, ensuring version control and easy access. We then deployed the images to Cloud Run using the gcloud run deploy command, enabling a scalable, serverless deployment environment.

Here is a link to one of our Dockerfiles: https://github.com/Sunray0466/Project-MLOps/blob/main/dockerfiles/train.dockerfile. Docker allowed us to streamline workflows, ensuring consistency across local and cloud environments.



### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- question 16 fill here ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
> Answer:

In our project, we utilized several GCP services to streamline development and deployment.
Artifact Registry was used to store and manage container images, ensuring version control and seamless integration with other GCP tools
Cloud Build automated our CI/CD pipelines, allowing us to efficiently build, test, and deploy containerized applications.
Also, we used Secret Manager to handle sensitive information securely and manage API keys and other credentials.
We have also used Vertex AI for training, managing, and deploying machine learning models at scale.
We stored our datasets and static files (model files) in Cloud Storage.
Finally, we deployed our containerized FastAPI and Streamlit implementations using Cloud Run, a serverless platform that automatically scaled based on demand.
### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
> Answer:

We initially used GCP’s Compute Engine to run small sample Docker images, helping us get familiar with the commands and user interface. This gave us a foundational understanding of managing virtual machines (VMs) and running containerized applications. While we explored the Compute Engine’s capabilities, including GPU usage, we later shifted our focus to Vertex AI for running real machine learning experiments. Compute Engine provided flexibility and scalability, but Vertex AI proved to be more efficient for registering jobs and managing experimentation workflows. During our exploration, we utilized NVIDIA_TESLA_P4 GPUs paired with an n1-standard-8 machine type, which offered a balance of performance and cost for smaller tasks. Though we didn’t rely heavily on Compute Engine in our final workflows, it was instrumental in building our understanding of GCP infrastructure.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

[The overview of the all buckets](figures/overall_buckets.png)
[Models folder and dataset](figures/model_folder.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

[Overview of the registry](figures/overall_registry.png)
[Docker files for training purposes](figures/training_docker.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

[Cloud Build History](figures/cloud_build.png)
[Continued](figures/cloud_build_cont.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
> Answer:

Yes, we managed to train our model in the cloud using Vertex AI. Before transitioning to the cloud, we ensured everything worked correctly on our local setup. We integrated our GitHub repository with a trigger that automatically built a Docker image for our training script. To adapt the script for cloud training, we copied and modified another file specifically for this purpose. For secure access, we stored our credentials in Secret Manager, which we later integrated into the Vertex AI job.

We created a YAML configuration file to submit the Vertex AI job and replaced environment variables with the secrets stored in Secret Manager during job creation. While the setup was efficient and streamlined, we encountered a challenge with GPU acceleration. When building GPU-accelerated Docker images, we faced dependency issues that we couldn’t resolve within our timeline. As a result, we opted to proceed with a CPU-based setup for the training job.

Despite this limitation, Vertex AI allowed us to register jobs and run training experiments efficiently in the cloud. Than, we can successfully saved the trained model in the Cloud Storage.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
> Answer:

We managed to make an working api with the use of FastAPI for setup and uvicorn for hosting a local server for testing said api. Setting up the FastAPI was smooth sailing due to the code provided by the course, but trying to extend the functionality gave a lot of headaches as soo many thing kept going wrong. The FastAPI documentation provided great help but was still lacking in some aspect such as what exactly the "UploadFile" class is as an object as it also could be considered as a bytes at some instances. And then the uploaded image could also not be .read() again after it was read once which gave issue later when we were implementing the frontend.
In the end we managed to solve theses issues and made a classifier endpoint which receives one or multiple card images and then outputs the predicted card(s) as well as the top three predictions for each card images and their probabilities. We also utilised the FastAPI lifespan argument to make a sort of init where we could load our model and nessesary files once and be used throughout the runtime which made the code a lot faster.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
> Answer:

By utilizing the uvicorn we could easily host a local server to test our api for bugs and features. Adding the `--reload` argument made the whole testing a lot easier as we didn't need to re-host the server each time we made changes to the file which was a great help. After writing our api, we would build it into a docker image which then gets uploaded to our gcloud run which would deploy our api and we could then use `/docs` from FastAPI to make sure that it all works (from another computer).

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
> Answer:

For unit testing, we used pytest to test the FastAPI endpoints. We validated the root endpoint's response and tested the image classification endpoint by sending valid and invalid files. The valid test ensured the correct prediction and probabilities were returned, while the invalid test confirmed proper error handling for non-image files.
For load testing, we would use the Locust framework, which allows simulating many users interacting with the application. It's easy to get started with and integrates well into our CI/CD pipeline. Using Locust, we would create tests to simulate multiple requests to the /classify/ endpoint, helping identify performance bottlenecks and assess the service’s behavior under heavy load.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did not manage to implement monitoring in our application. However, monitoring would be particularly beneficial for our use case, which involves predicting playing cards from images. It would allow us to track metrics like model accuracy, inference time, and response latency, helping us identify when the model starts making incorrect predictions or experiences performance issues. Monitoring could also highlight edge cases, such as poor-quality images or unusual card orientations, which could be used to improve the model. Furthermore, by analyzing user interactions, we could refine the model and enhance the user experience. Alerts for issues like low confidence scores or repeated prediction failures would help ensure quick identification and resolution of problems, ensuring the reliability and longevity of the application.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
> Answer:

We ended up with $42.82 in credits during the project. The most expensive parts of the project were training our model with Vertex AI and storing data in the cloud. Vertex AI costs were primarily due to running multiple experiments and managing job workflows, while cloud storage expenses arose from storing datasets and other resources needed for the project.

Overall, working in the cloud was a seamless and exciting experience. The ability to easily configure and adjust services offered a high level of flexibility, allowing us to adapt to project requirements quickly. The scalability of cloud services made it convenient to handle resource-intensive tasks without worrying about hardware limitations. Additionally, the integration between GCP services significantly streamlined our workflow. While cost management is essential, especially for resource-heavy tasks, the benefits of efficiency, scalability, and ease of use made working in the cloud a worthwhile and enjoyable experience.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
> Answer:

We implemented a simple frontend for our API (<a href="https://frontend-474989323251.europe-west1.run.app/">https://frontend-474989323251.europe-west1.run.app/</a>). This had a lot of trouble in the beginning as it was "though" it was incompatible with our backend (until we realized that it also accepted bytes data) or that reading a "uploaded file" once would make it unable to be read again. After fixing this issue, we spend a lot of time to make the UI more appealing, which we in the end are satisfied with. We would then try to add API monitoring with the "prometheus-client" package but in the end only had time to implement it in the files and not on the cloud (`/metrics` is still available).

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
> Answer:

[Overall Architecture](figures/mlops_diagram.png)

We are beginning with development environment, where code changes are prepared and committed. Before each commit, pre-commit hooks ensure code quality and formatting. Once changes are pushed to GitHub, GitHub Actions are triggered to run automated tests, including tests for data.py, model.py, and API implementations, while also verifying code formatting with Ruff.
Upon successful testing, Cloud Build automatically builds a Docker image of the latest codebase and stores it in Artifact Registry. This image serves as the foundation for deploying both training and serving workflows. Training jobs are triggered on Vertex AI, where the training script retrieves data from Cloud Storage and accesses credentials securely stored in Secret Manager. After training, the model is saved back to Cloud Storage for further use. The training process utilized CPU resources due to dependency issues encountered with GPU-accelerated Docker images.
To manage hyperparameter tuning and logging, we used Weights & Biases (W&B) and Hydra, which is used for experimentation and configuration management. Meanwhile, DVC was configured for tracking data versioning within Cloud Storage.
Once the model was trained, it was deployed to Cloud Run, a serverless platform that hosted both our FastAPI and Streamlit implementations for serving and interacting with the model. The ONNX format was used to make the model compatible for use on local machines when needed.

Overall, we have used GCP’s scalability and integration while maintaining flexibility across the pipeline.

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

The challenges we faced throughout the project was issues with working directory, unit test which kept falling due to data being accessed differently between github and the gcloud, making the frontend and backend communicate with each other (both local and on the cloud) and all the other problems where the terminal was involved in some way or other. Out of these, the biggest challenge we faced would be the issues with working directory. Due to our code being spread out on multiple platforms such as local machines, gcloud bucket, docker, wandb etc. we would kept getting errors, because either our data wasn't available or our other python files were not found and the code crashes. This also affects docker building as it needs the correct location, gcloud as it uses buckets instead to acquire our data, getting the correct model for our code etc.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
> Answer:

- Student s250069 was primarily responsible for continuous integration and testing throughout the project.

I set up the initial continuous integration (CI) pipeline on the GitHub repository and progressively enhanced it by adding caching, multi-OS, and multi-version testing for Python and PyTorch. I integrated a linting step into the CI pipeline and configured pre-commit hooks to enforce code quality standards. Additionally, I created workflows that trigger CI runs when data changes or when updates are made to the model registry. In terms of testing, I developed unit tests for the data processing components and model construction, ensuring robust coverage of the pipeline. I calculated code coverage and integrated it into the GitHub Actions workflow to track progress. I also wrote API tests for our application and set up dedicated CI workflows to validate API functionality. This comprehensive approach to CI and testing significantly improved the reliability and maintainability of the project.

- Student s242540 was primarily responsible for creating docker images and setup cloud environment for training, deployment.

I created all services in the cloud to set them up for our experimentation and deployment. I created trigger to Github to create Docker images automatically. I setup Vertex AI yaml files and quota arrangements. I deployed our model, frontend and backend to cloud also. To sum up, I did everything related to cloud work.

3. Student s242539 was primarily responsible for development of pre-trained model, integration of Weights & Biases and training the data

I integrated the Resnet-18 pre-trained model to our project and trained last layers. I spend most of the time to integrate W&B to our project up and running. I worked on training the data, set up hyperparameters for the experimentation and seamlessly saved our model and the corresponding performance metrics to W&B.

4. Student s204133 was primarily responsible for integration of hydra, development of backend, frontend and production of ONNX for deployment

I integrated hydra to our project to work seamlessly. Most of my time is spent on developing ONNX model, Fast-API generation and frontend implementation. I greatly helped to deploy the developments on cloud.
