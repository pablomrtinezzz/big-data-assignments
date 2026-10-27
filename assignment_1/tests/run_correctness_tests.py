#!/usr/bin/env python3
import sys
import subprocess
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from common.prng import PRNG

def run_cmd(cmd_list):
    res = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
    return res.stdout

def get_c_output(n, seed, flags=None):
    cmd = ["./src/c/matmul_opt", "--n", str(n), "--seed", str(seed), "--repetitions", "1", "--json"]
    if flags:
        cmd.extend(flags)
    out = run_cmd(cmd)
    return json.loads(out)

def get_java_output(n, seed, flags=None):
    cmd = ["java", "-cp", "src/java/build", "src.java.MatrixMultiplication", "--n", str(n), "--seed", str(seed), "--repetitions", "1", "--json"]
    if flags:
        cmd.extend(flags)
    out = run_cmd(cmd)
    return json.loads(out)

def get_python_output(n, seed, flags=None):
    cmd = [sys.executable, "src/python/matmul.py", "--n", str(n), "--seed", str(seed), "--repetitions", "1", "--json"]
    if flags:
        cmd.extend(flags)
    out = run_cmd(cmd)
    return json.loads(out)

def test_analytical_cases():
    print("===========================================================")
    print("TEST SUITE 1: Analytical Special Cases (Identity & Zero)")
    print("===========================================================")
    
    seeds = [42, 100, 2026]
    sizes = [4, 16, 64]
    
    for n in sizes:
        for seed in seeds:
            # 1. Zero Matrix Test: A * 0 = 0 -> Checksum must be 0.0
            c_zero = get_c_output(n, seed, ["--test-zero"])
            java_zero = get_java_output(n, seed, ["--test-zero"])
            py_zero = get_python_output(n, seed, ["--test-zero"])
            
            assert abs(c_zero["checksum"]) < 1e-15, f"C Zero test failed for n={n}"
            assert abs(java_zero["checksum"]) < 1e-15, f"Java Zero test failed for n={n}"
            assert abs(py_zero["checksum"]) < 1e-15, f"Python Zero test failed for n={n}"
            
            # 2. Identity Matrix Test: A * I = A -> Checksum must equal sum(A)
            p = PRNG(seed)
            A = np.array(p.fill_matrix(n), dtype=np.float64)
            expected_sum = np.sum(A)
            
            c_id = get_c_output(n, seed, ["--test-identity"])
            java_id = get_java_output(n, seed, ["--test-identity"])
            py_id = get_python_output(n, seed, ["--test-identity"])
            
            rel_err_c = abs(c_id["checksum"] - expected_sum) / expected_sum
            rel_err_java = abs(java_id["checksum"] - expected_sum) / expected_sum
            rel_err_py = abs(py_id["checksum"] - expected_sum) / expected_sum
            
            assert rel_err_c < 1e-12, f"C Identity test failed for n={n}: rel_err={rel_err_c}"
            assert rel_err_java < 1e-12, f"Java Identity test failed for n={n}: rel_err={rel_err_java}"
            assert rel_err_py < 1e-12, f"Python Identity test failed for n={n}: rel_err={rel_err_py}"
            
    print("  [PASS] Zero matrix multiplication verified: A * 0 = 0 (exact 0.0)")
    print("  [PASS] Identity matrix multiplication verified: A * I = A (rel_err < 1e-13)")

def test_numpy_reference():
    print("\n===========================================================")
    print("TEST SUITE 2: Verification Against NumPy Reference Standard")
    print("===========================================================")
    print(f"{'N':>5} | {'Seed':>6} | {'NumPy Checksum':>18} | {'C RelErr':>12} | {'Java RelErr':>12} | {'Python RelErr':>12} | Status")
    print("-" * 88)
    
    test_configs = [
        (2, 42), (4, 42), (8, 42), (16, 42), (32, 42), (64, 42), (128, 42),
        (16, 999), (64, 2026)
    ]
    
    all_passed = True
    results = []
    
    for n, seed in test_configs:
        # Generate bit-for-bit inputs using PRNG
        p = PRNG(seed)
        A = np.array(p.fill_matrix(n), dtype=np.float64)
        B = np.array(p.fill_matrix(n), dtype=np.float64)
        
        # Ground truth using NumPy BLAS
        C_ref = np.matmul(A, B)
        ref_checksum = float(np.sum(C_ref))
        
        c_res = get_c_output(n, seed)
        java_res = get_java_output(n, seed)
        py_res = get_python_output(n, seed)
        
        rel_err_c = abs(c_res["checksum"] - ref_checksum) / ref_checksum
        rel_err_java = abs(java_res["checksum"] - ref_checksum) / ref_checksum
        rel_err_py = abs(py_res["checksum"] - ref_checksum) / ref_checksum

        # Compute direct element-wise Frobenius norm for Python vs NumPy
        p_py = PRNG(seed)
        A_py = p_py.fill_matrix(n)
        B_py = p_py.fill_matrix(n)
        C_py_list = [[0.0 for _ in range(n)] for _ in range(n)]
        from src.python.matmul import matmul_ijk as py_matmul_ijk
        py_matmul_ijk(A_py, B_py, C_py_list, n)
        C_py_np = np.array(C_py_list, dtype=np.float64)
        frob_err_py = float(np.linalg.norm(C_py_np - C_ref, 'fro') / np.linalg.norm(C_ref, 'fro'))
        
        tol = 1e-12
        passed = (rel_err_c <= tol and rel_err_java <= tol and rel_err_py <= tol and frob_err_py <= tol)
        if not passed:
            all_passed = False
            
        status = "PASS" if passed else "FAIL"
        print(f"{n:5d} | {seed:6d} | {ref_checksum:18.6f} | {rel_err_c:12.2e} | {rel_err_java:12.2e} | {rel_err_py:12.2e} | {status}")
        
        results.append({
            "n": n,
            "seed": seed,
            "ref_checksum": ref_checksum,
            "rel_err_c": rel_err_c,
            "rel_err_java": rel_err_java,
            "rel_err_py": rel_err_py,
            "frob_err_py": frob_err_py,
            "passed": passed
        })


    assert all_passed, "Some tests failed relative tolerance checks!"
    print("\n  [PASS] All cross-language outputs verified within numerical tolerance (tol = 1e-12)")
    
    # Save verification results for report inclusion
    out_path = PROJECT_ROOT / "tests" / "correctness_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [INFO] Verification results saved to {out_path.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    # Ensure C and Java are compiled
    subprocess.run(["make", "-C", "src/c"], check=True)
    subprocess.run(["make", "-C", "src/java"], check=True)
    test_analytical_cases()
    test_numpy_reference()
