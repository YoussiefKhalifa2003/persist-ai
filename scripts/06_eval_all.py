import subprocess
import sys

subprocess.run([sys.executable, "-m", "lumen", "eval", *sys.argv[1:]], check=False)
