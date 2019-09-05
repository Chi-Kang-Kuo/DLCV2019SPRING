Modified from TA's README.


# HW3 ― GAN, ACGAN and UDA
In this assignment, you are given datasets of human face and digit images. You will need to implement the models of both GAN and ACGAN for generating human face images, and the model of DANN for classifying digit images from different domains.

<p align="center">
  <img width="550" height="500" src="https://lh3.googleusercontent.com/RvJZ5ZP0sVOqQ2qW7vIRJTP3PoIFCWGLYxvtYAjBKA2pLZWsyUICoBW9v_ENV6EsO7RBNVe1IIA">
</p>

For more details, please click [this link](https://1drv.ms/p/s!AmVnxPwdjNF2gZtOUMO5HEEQqLB8Ew) to view the slides of HW3.

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `hw3_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/uc?export=download&id=1gbnGEMyLIsYdIoyUyVZjYK8MzQZs4e_V) and unzip the compressed file manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `hw3_data` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

### Evaluation
To evaluate your UDA models in Problems 3 and 4, you can run the evaluation script provided in the starter code by using the following command.
