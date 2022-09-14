# Overview
This is a corpus which contains a lot of projects. 
In the root directory of subfolder `all_projects`, 
there are many directories, each directory is a project. 
For example, a directory named 'simple' is a project. 

## Steps to set up project for data preprocessor to handle
### Step1
In config.json, the field `data` is set to 'simple' meaning that 
currently used project is 'simple'. 
If you want to change to other project, for example: 'log4j-trunk', 
please set field `data` in config.json to 'log4j-trunk'. 
### Step2
After setting, find Java class: environment.EnvSetUp, this is a Java class with main funciton. 
Just run this main function. The specified project will be put into a specific folder under `user-home-directory`. 
The [data preprocessor](https://github.com/GrowingCode/JavaCodePreProcess) will 
translate Java files in that specific folder under `user-home-directory` to 
tensors which can be directly handled by [deep learning model](https://github.com/GrowingCode/FrameTokenMemAtten). 






