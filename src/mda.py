import glob
import pandas as pd
path = 'https://raw.githubusercontent.com/pufarin/micro-cocome-statistics/tree/master/measurements/api_gateway_1_to_1_db_distributed/*.csv'

all_rec = glob.glob(path)
print(all_rec)
#for p in all_rec:
#    print(p)