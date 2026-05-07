import subprocess
import time
import psutil
import sys
import os


# DATA_DIR and OUTPUT_DIR are expected to be set by NeuroFLAME when launching
# edge containers. If not set, utils.py falls back to test_data/<site_name>
# and test_output/<job_id>/<site_name> relative to the repo root.
# Explicitly log what's in the environment for debugging.
print(f"DATA_DIR: {os.getenv('DATA_DIR', '(not set)')}")
print(f"OUTPUT_DIR: {os.getenv('OUTPUT_DIR', '(not set)')}")

print("Starting the shell script...")
subprocess.Popen(["/bin/bash", "/workspace/runKit/startup/start.sh"])


time.sleep(10)

print("Polling for nvflare process and printing process details for debugging...")
while True:
    process_found = False
    for proc in psutil.process_iter(attrs=['cmdline']):
        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
        if 'nvflare' in cmdline:
            process_found = True
            print("nvflare process is running...")
            break

    if process_found:
        time.sleep(10)
    else:
        print("nvflare process is not running anymore or not found. Exiting.")
        sys.exit(0)
