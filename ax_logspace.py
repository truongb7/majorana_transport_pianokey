import numpy as np
import sys as sy

val_start = int(sy.argv[1])
val_end = int(sy.argv[2])
nbr_vals = int(sy.argv[3])

arr_logspace = np.round(np.logspace(val_start, val_end, num=nbr_vals), 6)
print(arr_logspace)
