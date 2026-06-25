import platform
import pathlib

if platform.system() == 'Windows':
    okeon_bucket = pathlib.Path(r"Y:\danielmk\birdcalledgeraw")
elif platform.system() == 'Linux':
    okeon_bucket = pathlib.Path("/bucket/FukaiU/danielmk/birdcalledgeraw/")

colors = ['#C70019', '#0D6B9A', '#EE9A20', '#6389A5', '#EA521C', '#8A963F']
