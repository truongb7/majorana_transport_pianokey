import numpy as np
import sys as sy

val_start = float(sy.argv[1])
val_end = float(sy.argv[2])
nbr_vals = int(sy.argv[3])

arr_geomspace = np.round(np.geomspace(val_start, val_end, num=nbr_vals), 6)
print(arr_geomspace)
