import subprocess
import sys

subprocess.run([sys.executable, "-m", "lumen", "track", "--method", "lumen", *sys.argv[1:]], check=False)
