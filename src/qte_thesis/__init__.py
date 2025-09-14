import os, math
from concurrent.futures import Executor
from pytask_parallel import ParallelBackend, registry
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

def _dask_executor(n_workers: int) -> Executor:
    cores = 96                               # per node (from sinfo)
    processes = min(cores, max(1, n_workers))
    jobs = max(1, math.ceil(n_workers / processes))

    cluster = SLURMCluster(
        queue="intelsr_short",
        account="ag_ifs_noack",
        cores=cores,
        processes=processes,
        threads=1,                           # 1 thread per process
        memory="180GB",
        job_extra_directives=["--nodes=1"],
        job_script_prologue=[
            'conda activate qte_thesis',
            'cd "$HOME/qte-thesis"',
            'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1'
        ],
    )
    cluster.scale(jobs=jobs)
    return Client(cluster).get_executor()

registry.register_parallel_backend(ParallelBackend.DASK, _dask_executor)