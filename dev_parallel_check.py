import sys
import subprocess
import shlex
from concurrent.futures import ThreadPoolExecutor
import shlex

def run_cmd(cmd):
    args = shlex.split(cmd)
    res = subprocess.run(args, shell=False, capture_output=True, text=True)
    return cmd, res.returncode, res.stdout, res.stderr

def run_parallel_checks():
    tasks = [
        "pytest --quiet",
        "git status --porcelain"
    ]
    results = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {executor.submit(run_cmd, task): task for task in tasks}
        for future in futures:
            cmd, code, out, err = future.result()
            results[cmd] = {"code": code, "stdout": out.strip(), "stderr": err.strip()}
    return results

if __name__ == '__main__':
    res = run_parallel_checks()
    for cmd, info in res.items():
        status = "PASSED" if info["code"] == 0 else "FAILED"
        print(f"[{status}] {cmd}")
    sys.exit(0 if all(v["code"] == 0 for v in res.values()) else 1)
