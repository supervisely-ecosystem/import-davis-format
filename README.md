<div align="center" markdown>
<img src="https://i.imgur.com/DTYFKGA.png"/>





# Import DAVIS

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Preparation">Preparation</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a>
</p>


[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/import-davis-format)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/import-davis-format.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/import-davis-format.png)](https://supervisely.com)

</div>

## Overview

The standalone DAVIS initiative is in maintenance mode: it won't be hosting any more and will no longer update. The DAVIS dataset is now part of the [RobMOTS Challenge](https://eval.vision.rwth-aachen.de/rvsu-workshop21/?page_id=110).

- [DAVIS 2016:](https://davischallenge.org/davis2016/code.html) In each video sequence a single instance is annotated.
- [DAVIS 2017 Semi-supervised:](https://davischallenge.org/davis2017/code.html#semisupervised) In each video sequence multiple instances are annotated.
- [DAVIS 2017 Unsupervised:](https://davischallenge.org/davis2017/code.html#unsupervised) In each video sequence multiple instances are annotated.

Semi-supervised and Unsupervised refer to the level of human interaction at test time, not during the training phase. In Semi-supervised, better called human guided, the segmentation mask for the objects of interest is provided in the first frame. In Unsupervised, better called human non-guided, no human input is provided.

App downloads data from official [DAVIS](https://davischallenge.org/). After extraction data is converted to [Supervisely](https://app.supervisely.com) format. 

## How To Run 

**Step 1**: Add app to your team from [Ecosystem](https://ecosystem.supervisely.com/apps/import-davis-format) if it is not there.

**Step 2**:  Open `Plugins & Apps` -> `Import DAVIS` -> `Run` 

<img src="https://i.imgur.com/688yito.png"/>

**Step 3**: Select the resolution and type of import of datasets.

<img src="https://i.imgur.com/oXaR7K5.png" width="800px"/>

Press `RUN`button. Now the window with program running logs will aappear. You don't have to wait for the program to finish execution(you can safely close the window).



## How to use

Resulting project will be placed to your current `Workspace`. Videos in datasets will have tags (`train`, `val`, `test`) corresponding to the input data.

<img src="https://i.imgur.com/mMj35R2.png"/>