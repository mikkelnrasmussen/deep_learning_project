# Deep learning project 2022 - Generative modelling for phenotypic profiling using Variational Autoencoders
This is a repository for the final project in Deep Learning fall 2022 at the technical university of Denmark (DTU). 

### The `data` directory:
The whole preprocessed dataset utilized in this project is available on the high-performance cluster (HPC) owned by DTU (G-bar) in the directory `/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/`. The original dataset is available from the Broad
Bioimage Benchmark Collection (BBBC) at https://bbbc.broadinstitute.org/BBBC021. A 1000 randomly sampled preprocessed single-cell images can be found in the `data` directory.

### The `code` directory:
As the dataset utilized in this project was too large to upload to GitHub or download to a personal laptop, minimal examples provided in the `code` folder, which are named:
- `minimal_example_standard_VAE.ipynb`
- `minimal_example_VAE_plus.ipynb`
- `minimal_example_VAE_classifier_model.ipynb`
- `minimal_example_images_classifier_model.ipynb`

which respectively presents the code utilized to train the standard VAE annd VAE+ model as well as the classifier model utilizing the VAE/VAE+ latent representationns of the single-cell images and the classifier model utilizing the single-cell images directly as input. In these minimal examples a 1000 randomly sampled images from the whole dataset are used to train the VAE, VAE+ and classifier models. Therefore the performance is not representative of the acutual perforamnce obtained when training on the whole dataset.

The VAE, VAE+ and classifier models were trained on GPUs using the queue system (bsub) at the HPC. The scripts `submit_VAE.sh`, `submit_VAE_plus.sh` and `submit_classifier` were respectively used to submit the model for training. These scripts are found in the `code` directory. The .sh script are only used to specify the GPU requirements used on the HPC and call the main python files. 

The pytorch implementation of the models can be found in the files `train_VAE.py`, `train_VAE_plus.py`, `train_classification_algo.py`, and `train_plain_classifier`. Further, the model architectures are stored inthe `model_VAE_plus.py`. Function used for plotting were stored in the `plotting.py` file. Finally, the scripts used for calculating the accuracies `calculate_train_accuracy.py` and `calculate_test_accuracy.py` are also provided.

### The `results` directory:
The results obtained when using training the VAE, VAE+ and classifier models on the HPC are provided in the `results` directory. However, as the files containing the final models were too large to upload to GitHub, these are not present. This also goes for the progression mpg4 videos created during the training procedure.

### The `visualize` directory:
All scripts utilized to create the visualization for report and evaluate the model performances are provided in the `virualize` directory. A handful of different visualization can also be found in the `visualize/results_visualizations`.
