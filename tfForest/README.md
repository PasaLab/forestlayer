## tfForest

### tabular data (ADULT, LETTER, YEAST, IMDB)
To run tfForest for datasets without spatial or sequential relationships, you need to perform following steps:

1. define your computing nodes (ip:port) in `run_codegen.py`, to generate running scripts for `ps` and `worker` respectively.

There are two types of running scripts:
``` shell
# ps.sh
python trainer.py --ps_hosts=192.168.100.1:3333
                  --worker_hosts=192.168.100.1:3334,192.168.100.2:3334,...192.168.100.16:3334
                  --job_name=ps --task_index=0 --data $1 --numSplit 2
```

``` shell
# worker.sh
python trainer.py --ps_hosts=192.168.100.1:3333
                  --worker_hosts=192.168.100.1:3334,192.168.100.2:3334,...192.168.100.16:3334
                  --job_name=worker --task_index=$1 --data $2 --numSplit 2
```

2. Then you need to run above two shell scripts (`ps.sh` and `worker.sh`) on corresponding computing nodes.
Taking the above cluster configuration as an example, you should run `ps.sh` on computing node `192.168.100.1`,
 and `sh worker.sh i <data_name>` on i-th worker (`192.168.100.<i+1>`) for i ranges from 0 to 15. 
Note that in this example, `ps` and one `worker` are started on `192.168.100.1`.

### image and sequence data (CIFAR10, MNIST, sEMG)

To run tfForest for datasets with spatial or sequential relationships, you should add a multi-grained scanning procedure.

Concretely, modify `ps.sh` and `worker.sh` stated above, change `trainer.py` to `mgscanner.py`. The cluster can be re-configured according the number of machines you want to used (i.e., using 12 machines).

After multi-grained scanning was performed, some intermidiate results will be generated. Then you can follow the instructions on handling tabular data above to perform a cascade forest procedure for these datasets.

### key files

* `load_<data_name>.py` : script that loads datasets. Note that you can create an directory `../DeepForestTF_Data` and put dataset into it.
* `kfoldwrapper.py` : wraping k-fold cross validation into class KFoldWrapper.
* `hash_23`, `hash_23mgs` : define hash value mapping on python3 using python2 hash function, to enable experiment re-production.
* `deepforestTF_new.ipynb` : you can further learn how tfForest works by this single-machine demo of jupyter notebook.
* `run_codegen.py` : generate scripts that start up the cluster (including parameter server and workers).
* `trainer.py` : define main function, define main work procedure of parameter server and workers. 
* `trainer_with_merger_tf.py` : you can add a new middle-layer which is called merger. merger can merge results of workers which are assigned parts of the same forest. By default we use no merger.



