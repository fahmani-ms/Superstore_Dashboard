import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import mysql.connector
import datetime
import math
from scipy import stats
from scipy.stats import f_oneway
from sklearn.preprocessing import MinMaxScaler
import mysql.connector
from scipy.stats import ttest_ind
import csv
username = 'root'
user_password = 'Sadegh74'
db_name = 'superstore'
cnx = mysql.connector.connect(
    user=username,
    password=user_password,
    host='localhost',
    database=db_name
)
query = "SELECT * FROM order_detail;"
df = pd.read_sql(query, con=cnx)
df.head()
w_dis = df[df['Discount'] != 0]
wo_dis = df[df['Discount'] == 0]
t_statistic, p_value = stats.ttest_ind(w_dis['Quantity'], wo_dis['Quantity'])

alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")
    
output = {}
output['No. Discounted'] = len(w_dis)
output['No. Without Discount'] = len(wo_dis)
output['hypothesis'] = 'The average quantity of discounted orders is same to the average quantity of non-discounted orders.'
output['t_statistic'] = t_statistic
output['p_value'] = p_value
output['alpha'] = alpha
output['result'] = 'Reject the Hypothesis' if p_value < alpha else 'Fail to Reject the Hypothesis'
output = pd.DataFrame(output, index=[0])