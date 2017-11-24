# Fast.ai Lesson 1 Project

This document contains a guide to help you to start experimenting with Fast.ai [Lesson 1 project](http://course.fast.ai/lessons/lesson1.html) **as easily as possible** through a Jupyter notebook running on FloydHub.

The code in this project was ported from the official course [repository](https://github.com/fastai/courses/tree/master/deeplearning1/nbs), but only the necessary files to run this specific project were selected. The data used in this project is already available as a FloydHub [dataset](https://www.floydhub.com/fastai/datasets/cats-vs-dogs).


Follow the next steps to get a jupyter notebook up and running for Lesson 1:

### 1. Create a FloydHub account

* [Sign up](https://www.floydhub.com/signup) on FloydHub
* Install the floyd CLI on your local machine through these two [steps](https://www.floydhub.com/welcome):

```
$ pip install -U floyd-cli

$ floyd login
# Follow the instructions on your CLI
```

### 2. Clone this project to your local machine

```
$ cd /path/to/your-project-dir
$ floyd clone fastai/projects/lesson1_dogs_cats/13
```

### 3. Create your project version on FloydHub
* [Create a project](https://www.floydhub.com/projects/create) on FloydHub and then sync the cloned repository with your new project
```
$ floyd init your-project-name
```


### 4. Run the project through a jupyter notebook

* The `--env` flag specifies the environment that this project should run on, which is a theano backend environment with Python 2
* The `--data` flag specifies that the cats-vs_dogs dataset should be available at the `/data` directory and the fast-ai-models dataset should be available at the `/models` directory
* The `--mode` flag specifies that this job should provide us a Jupyter notebook
* Note that the `--gpu` flag is optional for now, unless you want to start right away to run the code on a GPU machine. Since you'll be exploring and playing around with the code, you might not need a GPU instance right away, so you can avoid that flag now and restart it later with a GPU instance.

```
floyd run \
  --env theano-0.8:py2 \
  --data fastai/datasets/cats-vs-dogs/2:data \
  --data fastai/datasets/fast-ai-models/1:models \
  --mode jupyter \
  --gpu
```


Once the job is started, the jupyter notebook will open in your browser and you are ready to go!

### More about FloydHub platform

* Note that changes you make to the notebook running on Floyd servers are not going to be saved at your local directory, but they are going to be available in the [output](https://www.floydhub.com/fastai/projects/lesson1_dogs_cats/13/output) section of your job.
* Once you stop the Jupyter notebook job you were running, from the job page, you can click on [Restart](http://blog.floydhub.com/restart-jupyter-notebook-workflow/?utm_medium=email&utm_source=21sep17) to restart your job exactly how you left it and optionally choosing another environment (CPU or GPU). This is a great way to spend less of your GPU hours if you are working on tasks that does not require a GPU.
* If you need any help check our [documentation](http://docs.floydhub.com/) and [forum](https://forum.floydhub.com/).


### More datasets for Lesson 1

* If you want to exercise with different datasets, explore the [datasets](https://www.floydhub.com/fastai/datasets) uploaded specifically for the course