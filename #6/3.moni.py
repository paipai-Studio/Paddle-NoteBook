import warnings
warnings.filterwarnings("ignore")

import psutil
from pynvml import *
import subprocess
import time


class ResourceMonitor:
    def __init__(self):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)
        
    def get_stats(self):
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        gpu = nvmlDeviceGetUtilizationRates(self.handle).gpu
        return {"CPU": cpu, "Mem": mem, "GPU": gpu}


monitor = ResourceMonitor()
process = subprocess.Popen("python 2.run.py > 2.run.log", shell=True)


try:
    while True:
        stats = monitor.get_stats()
        print(f"CPU:\t{stats['CPU']:.2f}%\t|Mem:\t{stats['Mem']:.2f}%\t|GPU:\t{stats['GPU']:.2f}%")
        if process.poll() is not None: break
        time.sleep(60)
finally:
    process.terminate()
    nvmlShutdown()

