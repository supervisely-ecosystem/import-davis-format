<div align="center" markdown>
<img src="https://i.imgur.com/WJcTcJc.png" width="1900px"/>



# Import DAVIS

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Preparation">Preparation</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a>
</p>


[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/import-davis-format)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/import-davis-format&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/import-davis-format&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/import-davis-format&counter=runs&label=runs&123)](https://supervise.ly)

</div>

## Overview

App downloads `DAVIS` data from official `DAVIS` site([click here](https://davischallenge.org/davis2017/code.html#unsupervised)). After extraction data is converted to [Supervisely](https://app.supervise.ly) format. 

## How To Run 

**Step 1**: Add app to your team from [Ecosystem](https://ecosystem.supervise.ly/apps/import-cityscapes) if it is not there.

**Step 2**:  Open `Plugins & Apps` -> `Import DAVIS` -> `Run` 

<img src="https://i.imgur.com/pxFNPPh.png"/>

**Step 3**: Select the resolution and type of import of datasets.

<img src="https://i.imgur.com/vBW90az.png" width="800px"/>

Press `RUN`button. Now the window with program running logs will aappear. You don't have to wait for the program to finish execution(you can safely close the window).



## How to use

Resulting project will be placed to your current `Workspace` with name `davis2017` . Videos in datasets will have tags (`train`, `val`, `test`) corresponding to the input data.

<img src="https://i.imgur.com/mMj35R2.png"/>