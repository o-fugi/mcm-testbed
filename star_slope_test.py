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
    if(value > min_review_num):
        names_of_products_filtered.append(key)

percent_correct_sum = 0
num_products_filtered = 0
num_products_total = 0

rows_list = []

pd.set_option('display.max_rows', None)

# names_of_products_filtered = ['revlon 1875w volumizing hair dryer']

## Cycle through each of the products
for distance_to_shift_back in range(1, 15, 2):
    print(distance_to_shift_back)
    distance_to_shift_forward = distance_to_shift_back
    for idx, name in zip(range(0, len(names_of_products_filtered)), names_of_products_filtered):
        print(groupby_date, idx, len(names_of_products_filtered))
        df_product = df[df['product_title'] == name]
        df_product = df_product.sort_values(by=['review_date'])

        df_product['review_date_num'] = matplotlib.dates.date2num(df_product['review_date'])

        if(plot):
            df_product['review_date_num'] = df_product['review_date_num'] - 735000
            ax = df_product[['review_date_num', 'star_rating']].plot.scatter(x='review_date_num', y='star_rating')
            ax.set_ylabel('Stars')
            ax.set_xlabel('Days')
            ax.set_title('Unfiltered Star Ratings')
            plt.show()

        df_product['review_date_num'] = df_product['review_date_num'].astype('int32') // groupby_date # within two months 
        df_product = df_product.groupby('review_date_num').mean()
        df_product['review_date_num'] = df_product.index

        if(plot):
            df_product[['review_date_num', 'star_rating']].plot(x='review_date_num', y='star_rating')
            plt.show()

        if(len(df_product) < distance_to_shift_back + distance_to_shift_forward + rolling_window + 2):
            continue

        df_product['moving_average_stars'] = df_product['star_rating'].rolling(window=rolling_window).mean()

        df_product['slopes_back'] = (df_product['moving_average_stars'] - df_product.moving_average_stars.shift(1)) / (df_product['review_date_num'] - df_product.review_date_num.shift(1))
        for shift in range(2, distance_to_shift_back + 1):
            df_product['slopes_back'] += (df_product['moving_average_stars'] - df_product.moving_average_stars.shift(shift)) / (df_product['review_date_num'] - df_product.review_date_num.shift(shift))

        df_product['slopes_forward'] = (df_product['moving_average_stars'] - df_product.moving_average_stars.shift(-1)) / (df_product['review_date_num'] - df_product.review_date_num.shift(-1))
        for shift in range(2, distance_to_shift_forward + 1):
            df_product['slopes_forward'] += (df_product['moving_average_stars'] - df_product.moving_average_stars.shift(-shift)) / (df_product['review_date_num'] - df_product.review_date_num.shift(-shift))

        df_product['slopes_back'] = df_product['slopes_back'] / distance_to_shift_back
        df_product['slopes_forward'] = df_product['slopes_forward'] / distance_to_shift_forward
        df_product['significant_slopes'] = (abs(df_product['slopes_back']) > significant_slope_cutoff) & (abs(df_product['slopes_forward']) > significant_slope_cutoff)
        df_product['neither_significant_slopes'] = (abs(df_product['slopes_back']) < significant_slope_cutoff) & (abs(df_product['slopes_forward']) < significant_slope_cutoff)
        df_product['posneg_slopes_back'] = ~((df_product['slopes_back'] > 0) ^ (df_product['slopes_forward'] > 0)) & df_product['significant_slopes']

        if(plot):
            df_product.index = df_product['review_date_num'] - 52440
            ax = df_product[['moving_average_stars', 'star_rating']].loc[df_product['moving_average_stars'] == df_product['moving_average_stars']].plot()
            ax.set_ylabel('Stars')
            ax.set_xlabel('Days')
            ax.set_title('Moving Average of Star Rating')
            plt.show()

        df_product_filtered = df_product[df_product['slopes_back'] == df_product['slopes_back']]
        df_product_filtered = df_product_filtered[df_product_filtered['slopes_forward'] == df_product_filtered['slopes_forward']]
        df_product_filtered = df_product_filtered[df_product_filtered['significant_slopes']]
        if(len(df_product_filtered) > 0):
            # percent_correct = len(df_product_filtered[df_product_filtered['posneg_slopes_back'] | (df_product_filtered['neither_significant_slopes'])]) / len(df_product_filtered) # it's correct when it estimates zero or the correct direction
            percent_correct = len(df_product_filtered[df_product_filtered['posneg_slopes_back']]) / len(df_product_filtered) # it's correct when it estimates zero or the correct direction
            print(percent_correct, len(df_product_filtered))
            percent_correct_sum += percent_correct * len(df_product_filtered)
            num_products_filtered += len(df_product_filtered)
            num_products_total += 1

    dict1 = {}
    dict1.update({'distance_to_shift':distance_to_shift_back,'percent_correct': (percent_correct_sum / num_products_filtered), 'percent_appraised':num_products_total / len(names_of_products_filtered)})
    rows_list.append(dict1)
    percent_correct_sum = 0
    num_products_filtered = 0
    num_products_total = 0

df_cheat = pd.DataFrame(rows_list)
print(df_cheat)
print(df_cheat.loc[df_cheat.index == df_cheat.idxmax()['percent_correct']])

df_cheat.index = df_cheat['distance_to_shift']
ax = df_cheat[['percent_correct', 'percent_appraised']].plot()
ax.set_ylabel('Percent')
ax.set_xlabel('Distance to Look Forward/Back')
ax.set_title('Grouping Effectiveness')
plt.show()
# print(num_products_filtered)
# print(percent_correct_sum / num_products_filtered)
