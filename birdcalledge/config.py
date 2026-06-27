import platform
import pathlib

if platform.system() == 'Windows':
    okeon_bucket = pathlib.Path(r"Y:\danielmk\birdcalledgeraw")
elif platform.system() == 'Linux':
    okeon_bucket = pathlib.Path("/bucket/FukaiU/danielmk/birdcalledgeraw/")

colors_oist = ['#C70019', '#0D6B9A', '#EE9A20', '#6389A5', '#EA521C', '#8A963F']

colors_diverging = ['#b2182b', '#ef8a62', '#fddbc7', '#d1e5f0', '#67a9cf', '#2166ac']

colors_qual = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']

