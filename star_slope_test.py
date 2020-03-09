import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from statistics import mean
import matplotlib
import numpy as np

plot = False

## Set constants
distance_to_shift_back = 8
distance_to_shift_forward = 8
rolling_window = 20
significant_slope_cutoff = 0.03
groupby_date = 14
min_review_num = distance_to_shift_back + distance_to_shift_forward + rolling_window + 1
## Read data
# microwave, pacifier, hair_dryer
df = pd.read_csv('hair_dryer.tsv', sep='\t')
# Convert dates to datetime
df['review_date'] = pd.to_datetime(df['review_date'], infer_datetime_format=True)
df = df[['product_title', 'star_rating', 'review_date']]

## Filter the dataframe based on products with many reviews
names_of_products = df['product_title'].tolist()
dict_names_count = Counter(names_of_products)
dict_names_count_filtered = dict()
names_of_products_filtered = []
for (key, value) in dict_names_count.items():
    print(key, value)
    if(value > min_review_num):
        names_of_products_filtered.append(key)

percent_correct_sum = 0
num_products_filtered = 0

rows_list = []

pd.set_option('display.max_rows', None)

# names_of_products_filtered = ['revlon 1875w volumizing hair dryer']

## Cycle through each of the products
for idx, name in zip(range(0, len(names_of_products_filtered)), names_of_products_filtered):
    print(idx, name, len(names_of_products_filtered))
    df_product = df[df['product_title'] == name]
    df_product = df_product.sort_values(by=['review_date'])

    df_product['review_date_num'] = matplotlib.dates.date2num(df_product['review_date'])

    df_product['review_date_num'] = df_product['review_date_num'].astype('int32') // groupby_date # within two months 
    df_product = df_product.groupby('review_date_num').mean()
    df_product['review_date_num'] = df_product.index

    if(plot):
        df_product[['review_date_num', 'star_rating']].plot(x='review_date_num', y='star_rating')
        plt.show()

    if(len(df_product) < distance_to_shift_back + distance_to_shift_forward + rolling_window + 1):
        continue

    df_product['sma_stars'] = df_product['star_rating'].rolling(window=rolling_window).mean()

    df_product['slopes_back'] = (df_product['sma_stars'] - df_product.sma_stars.shift(1)) / (df_product['review_date_num'] - df_product.review_date_num.shift(1))
    for shift in range(2, distance_to_shift_back + 1):
        df_product['slopes_back'] += (df_product['sma_stars'] - df_product.sma_stars.shift(shift)) / (df_product['review_date_num'] - df_product.review_date_num.shift(shift))

    df_product['slopes_forward'] = (df_product['sma_stars'] - df_product.sma_stars.shift(-1)) / (df_product['review_date_num'] - df_product.review_date_num.shift(-1))
    for shift in range(2, distance_to_shift_forward + 1):
        df_product['slopes_forward'] += (df_product['sma_stars'] - df_product.sma_stars.shift(-shift)) / (df_product['review_date_num'] - df_product.review_date_num.shift(-shift))

    df_product['slopes_back'] = df_product['slopes_back'] / distance_to_shift_back
    df_product['slopes_forward'] = df_product['slopes_forward'] / distance_to_shift_forward
    df_product['significant_slopes'] = (abs(df_product['slopes_back']) > significant_slope_cutoff) & (abs(df_product['slopes_forward']) > significant_slope_cutoff)
    df_product['neither_significant_slopes'] = (abs(df_product['slopes_back']) < significant_slope_cutoff) & (abs(df_product['slopes_forward']) < significant_slope_cutoff)
    df_product['posneg_slopes_back'] = ~((df_product['slopes_back'] > 0) ^ (df_product['slopes_forward'] > 0)) & df_product['significant_slopes']

    if(plot):
        df_product.index = df_product['review_date_num']
        df_product[['sma_stars', 'star_rating', 'slopes_back', 'slopes_forward']].plot()
        plt.show()

    df_product_filtered = df_product[df_product['slopes_back'] == df_product['slopes_back']]
    df_product_filtered = df_product_filtered[df_product_filtered['slopes_forward'] == df_product_filtered['slopes_forward']]
    df_product_filtered = df_product_filtered[df_product_filtered['significant_slopes']]
    print(df_product_filtered)
    if(len(df_product_filtered) > 0):
        # percent_correct = len(df_product_filtered[df_product_filtered['posneg_slopes_back'] | (df_product_filtered['neither_significant_slopes'])]) / len(df_product_filtered) # it's correct when it estimates zero or the correct direction
        percent_correct = len(df_product_filtered[df_product_filtered['posneg_slopes_back']]) / len(df_product_filtered) # it's correct when it estimates zero or the correct direction
        print(percent_correct, len(df_product_filtered))
        percent_correct_sum += percent_correct * len(df_product_filtered)
        num_products_filtered += len(df_product_filtered)

# dict1 = {}
# dict1.update({'rolling_window':rolling_window,'percent_correct': (percent_correct_sum / num_products_filtered)})
# rows_list.append(dict1)
# percent_correct_sum = 0
# num_products_filtered = 0

# df_cheat = pd.DataFrame(rows_list)
# print(df_cheat)
# print(df_cheat.loc[df_cheat.index == df_cheat.idxmax()['percent_correct']])

print(percent_correct_sum / num_products_filtered)
print(num_products_filtered)
