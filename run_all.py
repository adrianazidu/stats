"""Run every module one after the other. Usage:  python run_all.py"""
import glob
import subprocess
import sys

for script in sorted(glob.glob("m*.py")):
    print("\n" + "#" * 70)
    print("#  " + script)
    print("#" * 70)
    subprocess.run([sys.executable, script])
