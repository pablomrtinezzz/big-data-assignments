"""
Distributed dense matrix multiplication with Apache Spark (PySpark RDD API).

Two partitioning strategies are implemented, both executing the same scalar
i-k-j C kernel (kernels/blockmm.c, reused from Assignment 3) on local blocks:

* ``bcast``   -- 1-D row-panel decomposition. A is split into P row panels that
                 are distributed as RDD partitions; B is shipped once to every
                 executor as a broadcast variable; each task computes one
                 independent panel of C. No shuffle.
* ``block2d`` -- 2-D block decomposition on a g x g grid (the replication/join
                 scheme used by Spark MLlib BlockMatrix.multiply). Block A(i,k)
                 is replicated to the g products (i,j,k), block B(k,j) to the g
                 products (i,j,k); the products are formed after a join
                 (shuffle) and partial results are summed with reduceByKey
                 (second shuffle) into C(i,j).

In both cases the driver owns A and B before the timed region and receives the
assembled C after it (explicit end-to-end boundary, see ``run_multiply``).
"""
from __future__ import annotations

import ctypes
import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIB_NAME = "libblockmm.dylib" if os.uname().sysname == "Darwin" else "libblockmm.so"
LIB_PATH = os.path.join(ROOT, "kernels", LIB_NAME)

# --------------------------------------------------------------------------
# Local kernel (runs inside Python workers on executors, or on the driver)
# --------------------------------------------------------------------------
_LIB = None
_DPTR = ctypes.POINTER(ctypes.c_double)


def _load_kernel():
    global _LIB
    if _LIB is None:
        path = LIB_PATH
        try:  # inside an executor: use the copy shipped with SparkContext.addFile
            from pyspark import SparkFiles
            cand = SparkFiles.get(LIB_NAME)
            if os.path.exists(cand):
                path = cand
        except Exception:
            pass
        lib = ctypes.CDLL(path)
        lib.block_ikj_acc.argtypes = [_DPTR, _DPTR, _DPTR,
                                      ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        lib.block_ikj_acc.restype = None
        _LIB = lib
    return _LIB


def block_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a @ b computed with the scalar i-k-j C kernel."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    c = np.zeros((m, n), dtype=np.float64)
    _load_kernel().block_ikj_acc(a.ctypes.data_as(_DPTR), b.ctypes.data_as(_DPTR),
                                 c.ctypes.data_as(_DPTR), m, k, n)
    return c


def bounds(n: int, parts: int):
    """Contiguous, near-equal split of range(n) into `parts` [lo, hi) intervals."""
    edges = np.linspace(0, n, parts + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(parts)]


# --------------------------------------------------------------------------
# Task functions (module level so they are pickled by reference)
# --------------------------------------------------------------------------
def _bcast_task(bB):
    def run(it):
        for idx, panel in it:
            t0 = time.perf_counter()
            b = bB.value                       # first access deserialises B in this worker
            t1 = time.perf_counter()
            c = block_matmul(panel, b)
            t2 = time.perf_counter()
            yield idx, c, t1 - t0, t2 - t1, os.getpid()
    return run


def _product(kv):
    (i, j, _k), (a, b) = kv
    t0 = time.perf_counter()
    c = block_matmul(a, b)
    return (i, j), (c, time.perf_counter() - t0, 1)


def _add(x, y):
    cx, tx, nx = x
    cy, ty, ny = y
    cx += cy                                  # in-place accumulation of partial blocks
    return cx, tx + ty, nx + ny


def _replicate_a(g):
    def f(kv):
        (i, k), blk = kv
        return [((i, j, k), blk) for j in range(g)]
    return f


def _replicate_b(g):
    def f(kv):
        (k, j), blk = kv
        return [((i, j, k), blk) for i in range(g)]
    return f


# --------------------------------------------------------------------------
# Distributed algorithms
# --------------------------------------------------------------------------
def multiply_bcast(sc, A: np.ndarray, B: np.ndarray, panels: int):
    n = A.shape[0]
    t = {}
    t0 = time.perf_counter()
    rb = bounds(n, panels)
    data = [(p, np.ascontiguousarray(A[lo:hi])) for p, (lo, hi) in enumerate(rb)]
    rdd = sc.parallelize(data, numSlices=panels)
    bB = sc.broadcast(B)
    t1 = time.perf_counter()
    out = rdd.mapPartitions(_bcast_task(bB)).collect()
    t2 = time.perf_counter()
    C = np.empty((n, n), dtype=np.float64)
    for p, c, _, _, _ in out:
        lo, hi = rb[p]
        C[lo:hi] = c
    t3 = time.perf_counter()
    bB.destroy()
    t.update(distribute=t1 - t0, job=t2 - t1, assemble=t3 - t2, e2e=t3 - t0,
             kernel_sum=sum(o[3] for o in out), kernel_max=max(o[3] for o in out),
             bcast_fetch_sum=sum(o[2] for o in out),
             tasks=len(out), worker_pids=len({o[4] for o in out}))
    return C, t


def multiply_block2d(sc, A: np.ndarray, B: np.ndarray, g: int):
    n = A.shape[0]
    t = {}
    t0 = time.perf_counter()
    rb = bounds(n, g)
    a_blocks = [((i, k), np.ascontiguousarray(A[rb[i][0]:rb[i][1], rb[k][0]:rb[k][1]]))
                for i in range(g) for k in range(g)]
    b_blocks = [((k, j), np.ascontiguousarray(B[rb[k][0]:rb[k][1], rb[j][0]:rb[j][1]]))
                for k in range(g) for j in range(g)]
    nparts = g * g
    ra = sc.parallelize(a_blocks, numSlices=nparts).flatMap(_replicate_a(g))
    rbk = sc.parallelize(b_blocks, numSlices=nparts).flatMap(_replicate_b(g))
    t1 = time.perf_counter()
    out = (ra.join(rbk, numPartitions=nparts)
             .map(_product)
             .reduceByKey(_add, numPartitions=nparts)
             .collect())
    t2 = time.perf_counter()
    C = np.empty((n, n), dtype=np.float64)
    for (i, j), (c, _, _) in out:
        C[rb[i][0]:rb[i][1], rb[j][0]:rb[j][1]] = c
    t3 = time.perf_counter()
    t.update(distribute=t1 - t0, job=t2 - t1, assemble=t3 - t2, e2e=t3 - t0,
             kernel_sum=sum(v[1] for _, v in out), kernel_max=float("nan"),
             bcast_fetch_sum=0.0, tasks=sum(v[2] for _, v in out), worker_pids=0)
    return C, t


STRATEGIES = {"bcast": multiply_bcast, "block2d": multiply_block2d}


# --------------------------------------------------------------------------
# Cluster session
# --------------------------------------------------------------------------
@dataclass
class ClusterConfig:
    workers: int                 # executor processes (one Spark worker each)
    cores: int = 1               # task slots per executor
    worker_mem_mb: int = 1024    # memory offered by each Spark worker
    executor_mem: str = "640m"   # JVM heap per executor
    driver_mem: str = "2g"

    @property
    def master(self) -> str:
        return f"local-cluster[{self.workers},{self.cores},{self.worker_mem_mb}]"

    @property
    def label(self) -> str:
        return f"{self.workers}x{self.cores}"


def start_session(cfg: ClusterConfig, app="A4-distributed-matmul"):
    from pyspark.sql import SparkSession
    single = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
              "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    t0 = time.perf_counter()
    b = (SparkSession.builder.master(cfg.master).appName(app)
         .config("spark.executor.memory", cfg.executor_mem)
         .config("spark.executor.cores", str(cfg.cores))
         .config("spark.driver.memory", cfg.driver_mem)
         .config("spark.driver.maxResultSize", "4g")
         .config("spark.python.worker.reuse", "true")
         .config("spark.ui.showConsoleProgress", "false")
         .config("spark.ui.enabled", "true")
         .config("spark.sql.shuffle.partitions", "16")
         .config("spark.dynamicAllocation.enabled", "false"))
    for k, v in single.items():
        b = b.config(f"spark.executorEnv.{k}", v)
        os.environ.setdefault(k, v)
    spark = b.getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    sc.addFile(LIB_PATH)
    sc.addPyFile(os.path.abspath(__file__))
    # wait until all executors registered (local-cluster starts them asynchronously)
    deadline = time.time() + 120
    while len(executors(sc)) < cfg.workers and time.time() < deadline:
        time.sleep(0.1)
    startup = time.perf_counter() - t0
    return spark, sc, startup


def _rest(sc, path):
    url = f"{sc.uiWebUrl}/api/v1/applications/{sc.applicationId}/{path}"
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def executors(sc):
    try:
        return [e for e in _rest(sc, "executors") if e["id"] != "driver"]
    except Exception:
        return []


def stage_metrics(sc, job_group: str):
    """Aggregate Spark task metrics (REST API) over all stages of a job group."""
    tracker = sc.statusTracker()
    stage_ids = set()
    for jid in tracker.getJobIdsForGroup(job_group):
        info = tracker.getJobInfo(jid)
        if info:
            stage_ids.update(info.stageIds)
    keys = ["executorRunTime", "executorCpuTime", "executorDeserializeTime",
            "resultSerializationTime", "jvmGcTime", "resultSize",
            "shuffleReadBytes", "shuffleWriteBytes", "shuffleFetchWaitTime",
            "shuffleWriteTime", "inputBytes", "numCompleteTasks"]
    agg = {k: 0 for k in keys}
    agg["stages"] = 0
    for sid in stage_ids:
        for _ in range(50):              # listener bus is asynchronous
            try:
                st = _rest(sc, f"stages/{sid}")
            except Exception:
                st = []
            st = [s for s in st if s.get("status") in ("COMPLETE", "SKIPPED")]
            if st:
                break
            time.sleep(0.1)
        for s in st:
            if s.get("status") != "COMPLETE":
                continue
            agg["stages"] += 1
            for k in keys:
                agg[k] += s.get(k, 0) or 0
    return agg


# --------------------------------------------------------------------------
# Memory sampler (psutil): peak RSS per executor process tree and driver
# --------------------------------------------------------------------------
class MemorySampler:
    """Samples RSS of every executor JVM subtree (JVM + Python daemon + Python
    workers) and of the driver (Python driver + driver JVM) every `period` s."""

    def __init__(self, period=0.02):
        import psutil
        self.psutil = psutil
        self.period = period
        self.me = psutil.Process()
        self._stop = threading.Event()
        self.peak_exec = {}
        self.peak_driver = 0
        self.samples = 0
        self._th = None

    def _snapshot(self):
        ps = self.psutil
        execs, driver = {}, self.me.memory_info().rss
        try:
            kids = self.me.children(recursive=False)
        except ps.Error:
            return execs, driver
        for jvm in kids:                                   # driver JVM
            try:
                driver += jvm.memory_info().rss
                for ch in jvm.children(recursive=False):  # executor JVMs
                    cmd = " ".join(ch.cmdline())
                    if "CoarseGrainedExecutorBackend" not in cmd:
                        continue
                    tot = ch.memory_info().rss
                    for d in ch.children(recursive=True):
                        try:
                            tot += d.memory_info().rss
                        except ps.Error:
                            pass
                    execs[ch.pid] = tot
            except ps.Error:
                continue
        return execs, driver

    def _loop(self):
        while not self._stop.is_set():
            execs, drv = self._snapshot()
            for pid, v in execs.items():
                self.peak_exec[pid] = max(self.peak_exec.get(pid, 0), v)
            self.peak_driver = max(self.peak_driver, drv)
            self.samples += 1
            self._stop.wait(self.period)

    def __enter__(self):
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join()

    def summary(self):
        v = list(self.peak_exec.values()) or [0]
        return {"exec_peak_rss_max": max(v), "exec_peak_rss_mean": float(np.mean(v)),
                "exec_peak_rss_sum": int(sum(v)), "driver_peak_rss": self.peak_driver,
                "mem_samples": self.samples, "exec_seen": len(self.peak_exec)}


def run_multiply(sc, strategy: str, A, B, param: int, tag: str, measure=True):
    """One timed multiplication. End-to-end boundary: A and B already in driver
    memory -> partitioning/serialisation -> distributed job -> C assembled in
    the driver. Spark session start-up is excluded (reported separately)."""
    sc.setJobGroup(tag, tag)
    if measure:
        with MemorySampler() as ms:
            C, t = STRATEGIES[strategy](sc, A, B, param)
        t.update(ms.summary())
        t.update(stage_metrics(sc, tag))
    else:
        C, t = STRATEGIES[strategy](sc, A, B, param)
    return C, t
