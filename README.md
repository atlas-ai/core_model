# Core Model

## Install

```bash
virtualenv venv --python python3
pip3 install -r requirements.txt
```

get the testdata repository cloned in parent repo:

```bash
cd ..
git clone git@github.com:atlas-ai/data.git
```

## Running

```bash
./Batch_Run.py
```

## Interactive use with Jupyter

###  Add the path of the `core_model` to your PYTHONPATH


```
vim ~/.bash_profile 
```
add at the end of the file

```
export PYTHONPATH=$PYTHONPATH:"/Users/my_user/path_to_the_code/core_model"
```

where `"/Users/my_user/path_to_the_code/core_model"` is the actual path to the code

Save the file (type `:wq` in vim)

The change will be efective for the new terminals you lanch. It need to be done only once


#### Launch jupyter

```
jupyter notebook
```