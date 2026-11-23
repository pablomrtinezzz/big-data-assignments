"""Record the hardware/software environment to results/environment.json."""
import json
import os
import platform
import subprocess


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def main():
    import numpy, pyspark, psutil
    env = {
        "cpu": sh("sysctl -n machdep.cpu.brand_string"),
        "physical_cores": int(sh("sysctl -n hw.physicalcpu") or 0),
        "logical_cpus": int(sh("sysctl -n hw.logicalcpu") or 0),
        "perf_cores": sh("sysctl -n hw.perflevel0.physicalcpu"),
        "eff_cores": sh("sysctl -n hw.perflevel1.physicalcpu"),
        "ram_bytes": int(sh("sysctl -n hw.memsize") or 0),
        "os": f"{platform.system()} {platform.release()} ({sh('sw_vers -productVersion')})",
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pyspark": pyspark.__version__,
        "psutil": psutil.__version__,
        "java": sh("java -version 2>&1 | head -1"),
        "clang": sh("clang --version | head -1"),
        "kernel_cflags": "-O3 -std=c99 -fPIC (vectorize(disable) on inner loop)",
        "a3_cflags": "-O3 -std=c99 -Xclang -fopenmp (libomp)",
        "machines": 1,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "environment.json")
    json.dump(env, open(out, "w"), indent=2)
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
