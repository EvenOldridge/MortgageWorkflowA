# MortgageWorkflowA
An initial workflow for tabular deep learning, tested on the mortgage dataset, to be used for profiling I/O and GPU/CPU Usage. 

Launch with an up to date version of the rapids container.  Additional libraries are pip installed from within the notebook.

> Docker run --runtime=nvidia -v /tmp/eoldridge/fnma_full_data_proc_out4/dnn/:/data/mortgage/ --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 rapidsai/rapidsai:latest
