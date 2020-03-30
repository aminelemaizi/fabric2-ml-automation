# Very Basic Machine Learning Automation Using Fabric 2
You can find a detailed blog article explaining this code here : https://lemaizi.com/blog/very-basic-machine-learning-automation-using-fabric-2/.

### Project Folder Tree

```
.
├── fabfile.py
├── assets
│   ├── secrets.json
│   └── server_ssh_key
└── ml_assets
    ├── mnist_train.csv
    ├── model.py
    └── Requirements.txt
```

### What does this code do ?

This code automate the training and the prediction of an Autoencoder. In any Ubuntu server (18.04 LTS)  we can run the training, get the model and generate autoencoded images to be retrieved to the local machine, by using this four commands :

```bash
# Install needed packages in remote OS
fab2 prepare-os

# Upload assets and create a Python virtual environment
fab2 prepare-menv

# Launch the training and retrieve the model to local assets
fab2 train-model --dataset=<link_to_train_data> --epochs=<nb_of_epochs>

# Run predictions
fab2 predict-data --dataset=<link_to_train_data>
```

### How to adapt the code to your needs ?

- First make sure you've installed [Fabric 2](http://www.fabfile.org/)
- Modify the `train` and `predict` functions inside `model.py`
- Adapt the Fabric 2 tasks inside `fabfile.py`
- Put you server SSH encryption key inside `assets` as `server_ssh_key`
- Put your sudo password and encryption key passphrase inside `secrets.json`
- Put your training dataset inside `ml_assets` 