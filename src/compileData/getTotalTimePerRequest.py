import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def millisecondsToSeconds(millis):
    seconds=(millis/1000)
    return round(seconds)

df1 = pd.read_csv('./pub_sub_1_to_1_db_distributed/SR_OP&ROP_DR_10_10_10.csv')
#df1 = df1.sort_values(by=['uuid'])

df2 = pd.read_csv('./pub_sub_1_to_1_db_distributed/incoming_10_10_10.csv')
#df2 = df2.sort_values(by=['uuid'])

#print(np.where(df1['uuid'] == df2['uuid'], 'True', 'False'))

df3 = df1.join(df2.set_index('uuid'), on='uuid')

#df3.to_csv('./pub_sub_1_to_1_db_distributed/joinedData.csv', encoding='utf-8', index=False)


df3['total_time'] = df3['time_received'] - df3['timeStamp']

df3.to_csv('./pub_sub_1_to_1_db_distributed/completeData.csv', encoding='utf-8', index=False)


# histogram
#sns_plot = sns.distplot( df["total"])
#sns_plot.figure.savefig("output1.png")

#print(df.describe())


