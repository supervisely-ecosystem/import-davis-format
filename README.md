<div align="center" markdown>
<img src="https://i.imgur.com/48JFbtt.png" width="1900px"/>


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

Import data in [DAVIS](https://davischallenge.org/) format to [Supervisely](https://supervise.ly/) from folder.

## Preparation

Download `DAVIS` data from official `DAVIS` site([click here](https://davischallenge.org/davis2017/code.html#unsupervised)). You need to download `DAVIS-2017` archive(`DAVIS-2017-trainval-480p.zip` or `DAVIS-2017-Unsupervised-trainval-480p.zip` or `DAVIS-2017-Unsupervised-trainval-Full-Resolution.zip`) with input images and annotations, and  `DAVIS-2017_semantics-480p.zip` - archive with `davis_semantics.json` file, containing the category for each object in annotations. 

Upload your data in `DAVIS` format to `Team Files` (for example you can create `import_davis` folder).

<img src="https://i.imgur.com/45uOaK0.png"/>

## How To Run 

**Step 1**: Add app to your team from [Ecosystem](https://ecosystem.supervise.ly/apps/import-cityscapes) if it is not there.

**Step 2**: Go to `Current Team`->`Files` page, right-click on your `folder`, containing `davis` data and choose `Run App`->`Import Davis`. You will be redirected to `Workspace`->`Tasks` page. 

<img src="https://i.imgur.com/dJr5sLz.png"/>

## How to use

Resulting project will be placed to your current `Workspace` with the same name as the `davis` archive. Videos in datasets will have tags (`train`, `val`) corresponding to the data in `DAVIS-2017` archive.

<img src="https://i.imgur.com/UC0ygAH.png"/>

You can also access your project by clicking on it's name from `Tasks` page.

<img src="https://i.imgur.com/h54uGur.png">