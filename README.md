required packages: torch numpy matplotlib palettable fire tqdm

```
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

to run variational inference in parameter space or function space
```
python run_reg_vi.py --dataset=sin --space=param
python run_reg_vi.py --dataset=sin --space=function_sampling
```

to run variational merging, note that each will try both param space and function space merging. We can specify what we want at the client level using the space_client arg, and how we want the data to be split using the homogenous arg.
```
python run_reg_merge_vi.py --dataset=sin --no_epochs_client=10000 --no_epochs_server=10000 --act_func=relu --space_client=function_sampling --network_size=[1,50,50,1] --homogeneous=False
python run_reg_merge_vi.py --dataset=sin --no_epochs_client=10000 --no_epochs_server=10000 --act_func=relu --space_client=function_sampling --network_size=[1,50,50,1] --homogeneous=True
python run_reg_merge_vi.py --dataset=sin --no_epochs_client=10000 --no_epochs_server=10000 --act_func=relu --space_client=param --network_size=[1,50,50,1] --homogeneous=False
python run_reg_merge_vi.py --dataset=sin --no_epochs_client=10000 --no_epochs_server=10000 --act_func=relu --space_client=param --network_size=[1,50,50,1] --homogeneous=True
```

The binary classification example is not tested yet. Parameter space works fine:
```
python run_bincla_vi.py --space=param
```

Each of these runs will save a couple of figs to /tmp/ showing the objective trajectory and the data + predictions.