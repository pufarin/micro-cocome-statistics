import pandas as pd
from glob import iglob, glob
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import operator as op


def get_data_from_csv(absolut_path, dir_name, column_names):
    path = r'{}/{}/*.csv'.format(absolut_path, dir_name)
    file_path = glob(path, recursive=True)
    print(file_path)
    return pd.read_csv(file_path[0], skipinitialspace=True, usecols=column_names)


def get_data_from_csv_files(absolut_path, dir_name, column_names):
    path = r'{}\{}\*.csv'.format(absolut_path, dir_name)
    return pd.concat((pd.read_csv(f, skipinitialspace=True, usecols=column_names) for f in iglob(path, recursive=True)),
                     ignore_index=True)


def remove_outliers(data_frame):
    z = np.abs(stats.zscore(data_frame['elapsed']))
    return data_frame[(z < 3)]


def remove_outliersPubSub(data_frame):
    z = np.abs(stats.zscore(data_frame['total_time']))
    return data_frame[(z < 3)]


def get_callback_dir(dir_name, call_back):
    return r'{}\{}'.format(dir_name, call_back)


absolutPath = r'D:\gabriel\Uni\Master\gitHub\micro-cocome-statistics\measurements'
callBack = r'callBackData'

ag1to1 = r'api_gateway_1_to_1_db_distributed'
agOne = r'api_gateway_one_db_distributed'
m1to1 = r'master_1_to_1_db_distributed'
mOne = r'master_one_db_distributed'
oAg1to1 = r'orchestrate_api_gateway_1_to_1_db_distributed'
ps1tot1 = r'pub_sub_1_to_1_db_distributed'
psOne = r'pub_sub_one_db_distributed'
mb1to1 = r'message_bus_1_to_1_db_distributed'
opb1to1 = r'orchestrate_pub_sub_1_to_1_db'

fields = ['elapsed']
fieldsAsync = ['timeStamp', 'uuid']
fieldsAsyncCallback = ['time_received', 'uuid']

sns.set(font_scale=2.5)

# api_gateway_1_to_1_db
apg1 = get_data_from_csv_files(absolutPath, ag1to1, fields)
prunedApg1 = remove_outliers(apg1)

# api_gateway_one_db
apgOne = get_data_from_csv_files(absolutPath, agOne, fields)
prunedApgOne = remove_outliers(apgOne)

# master_1_to_1_db_distributed
master1 = get_data_from_csv_files(absolutPath, m1to1, fields)
prunedMaster1 = remove_outliers(master1)

# master_one_db_distributed
masterOne = get_data_from_csv_files(absolutPath, mOne, fields)
prunedMasterOne = remove_outliers(masterOne)

# # orchestrate_api_gateway_1_to_1_db_distributed
oApi1to1 = get_data_from_csv_files(absolutPath, oAg1to1, fields)
prunedOApi1to1 = remove_outliers(oApi1to1)

# pub_sub_1_to_1_db_distributed
pubSub1to1 = get_data_from_csv_files(absolutPath, ps1tot1, fieldsAsync)
pubSub1to1Callback = get_data_from_csv(absolutPath, get_callback_dir(ps1tot1, callBack), fieldsAsyncCallback)

pubSub1to1Aggregate = pubSub1to1.join(pubSub1to1Callback.set_index('uuid'), on='uuid')
pubSub1to1Aggregate['total_time'] = pubSub1to1Aggregate['time_received'] - pubSub1to1Aggregate['timeStamp']
prunedPubSub1to1 = remove_outliersPubSub(pubSub1to1Aggregate)

# pub_sub_one_db_distributed
pubSubOne = get_data_from_csv_files(absolutPath, psOne, fieldsAsync)
pubSubOneCallback = get_data_from_csv(absolutPath, get_callback_dir(psOne, callBack), fieldsAsyncCallback)

pubSubOneAggregate = pubSubOne.join(pubSubOneCallback.set_index('uuid'), on='uuid')
pubSubOneAggregate['total_time'] = pubSubOneAggregate['time_received'] - pubSubOneAggregate['timeStamp']
prunedPubSubOne = remove_outliersPubSub(pubSubOneAggregate)

# message_bus_1_to_1_db_distributed

mbus1to1 = get_data_from_csv_files(absolutPath, mb1to1, fieldsAsync)
mbus1to1Callback = get_data_from_csv(absolutPath, get_callback_dir(mb1to1, callBack), fieldsAsyncCallback)

mbus1to1Aggregate = mbus1to1Callback.join(mbus1to1.set_index('uuid'), on='uuid')
mbus1to1Aggregate['total_time'] = mbus1to1Aggregate['time_received'] - mbus1to1Aggregate['timeStamp']
prunedMbus1to1 = remove_outliersPubSub(mbus1to1Aggregate)

# orchestrate_pub_sub_1_to_1_db

oPubSub1to1 = get_data_from_csv_files(absolutPath, opb1to1, fieldsAsync)
oPubSub1to1Callback = get_data_from_csv(absolutPath, get_callback_dir(opb1to1, callBack), fieldsAsyncCallback)

oPubSub1to1Aggregate = oPubSub1to1Callback.join(oPubSub1to1.set_index('uuid'), on='uuid')
oPubSub1to1Aggregate['total_time'] = oPubSub1to1Aggregate['time_received'] - oPubSub1to1Aggregate['timeStamp']
prunedOPubSub1to1 = remove_outliersPubSub(oPubSub1to1Aggregate)

# plots
# https://stackoverflow.com/questions/42004381/box-plot-of-a-many-pandas-dataframes
# https://stackoverflow.com/questions/43434020/black-and-white-boxplots-in-seaborn


data_dict = {'v01': prunedMaster1['elapsed'], 'v02': prunedMasterOne['elapsed'], 'v03': prunedApg1['elapsed'],
             'v04': prunedApgOne['elapsed'], 'v05': prunedOPubSub1to1['total_time'],
             'v06': prunedPubSubOne['total_time'],
             'v07': prunedMbus1to1['total_time'], 'v09': prunedOApi1to1['elapsed'],
             'v10': prunedOPubSub1to1['total_time']}

dd = pd.DataFrame(data=data_dict)

PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'blue'},
    'medianprops':{'color':'red'},
    'whiskerprops':{'color':'black'}
}

plt.rcParams['figure.figsize'] = (15.0, 12.0)
ax = sns.boxplot(data=dd, linewidth=2.5, **PROPS)  # RUN PLOT
plt.ylabel("Total Time", fontsize=35)
plt.xlabel("Versions", fontsize=35)
plt.xticks(size=30)
plt.yticks(size=30)
plt.savefig('BoxPlot', bbox_inches='tight')



