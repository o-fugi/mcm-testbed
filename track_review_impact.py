import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from statistics import mean
import numpy as np
import matplotlib

# TODO
# Instead of looking at the average star rating per day, look at individual reviews
# When looking at past reviews, take the weights of the reviews into account
# Look at average of previous reviews correlation with each review, instead of "last n"
# How can you tell if this is a statistically significant correlation?

## Set constants
min_review_num = 15
previous_reviews_to_calc = 10
groupby_date = 14

## Read data
# microwave, pacifier, hair_dryer
df = pd.read_csv('microwave.tsv', sep='\t')
# Convert dates to datetime
df['review_date'] = pd.to_datetime(df['review_date'], infer_datetime_format=True)

## Filter the dataframe based on products with many reviews
names_of_products = df['product_title'].tolist()
dict_names_count = Counter(names_of_products)
dict_names_count_filtered = dict()
names_of_products_filtered = []
for (key, value) in dict_names_count.items():
    if(value > min_review_num):
        names_of_products_filtered.append(key)

## Initialize the lists and dataframes to store correlation data
correlations = []
num_reviews = []
df_correlations = pd.DataFrame(columns = ['star_rating', 'previous_star_ratings'])

pd.set_option('display.max_rows', None)

## Cycle through each of the products
for name in names_of_products_filtered:
    print(name)
    plt.clf()
    df_product = df[df['product_title'] == name]
    df_product = df_product.sort_values(by=['review_date'])

    # np_product_arr = df_product.to_numpy()

    #messing with helpful votes stuff
    df_product['helpful_votes_rank'] = df['helpful_votes'].rank(ascending=True, method='min')
    df_product['weighted_star_rating'] = df_product['helpful_votes_rank'] * df_product['star_rating']

    df_product['review_date_num'] = matplotlib.dates.date2num(df_product['review_date'])

    df_product['review_date_num'] = df_product['review_date_num'].astype('int32') // groupby_date # within two months 
    df_grouped_product = df_product.groupby('review_date_num').mean()
    df_grouped_product['review_date_num'] = df_grouped_product.index

    star_list = df_grouped_product['star_rating'].tolist();
    previous_star_list = [sum(star_list[:i]) / i for i in range(1, len(star_list))]
    previous_star_list.insert(0, 0)
    df_grouped_product['all_previous_star_ratings'] = previous_star_list

    previous_star_reviews = df_grouped_product.star_rating.shift(1)
    for shift in range(1, previous_reviews_to_calc):
        previous_star_reviews += df_grouped_product.star_rating.shift(shift + 1)


    df_grouped_product['previous_star_ratings'] = previous_star_reviews
    df_grouped_product['previous_star_ratings'] = df_grouped_product['previous_star_ratings'] / previous_reviews_to_calc

    print(df_grouped_product)
    correlations.append(df_grouped_product.corr(method='pearson')['star_rating']['previous_star_ratings'])
    num_reviews.append(len(df_grouped_product.index))
    df_correlations = df_correlations.append(df_grouped_product[['star_rating', 'previous_star_ratings', 'all_previous_star_ratings']])


    df_grouped_product['sma_stars'] = df_grouped_product['star_rating'].rolling(window=10).mean()
    df_grouped_product[['star_rating', 'previous_star_ratings']].plot()
    plt.show()
        
# Let's fuck around with the data

n, bins, patches = plt.hist(correlations)
plt.show()

weighted_correlations = [ a * b for a, b in zip(correlations, num_reviews) ]
weighted_correlations = [x for x in weighted_correlations if str(x) != 'nan']

print(mean(weighted_correlations))

print(mean(df_correlations['star_rating'].tolist()))
df_correlations = df_correlations[df_correlations['star_rating'] == df_correlations['star_rating']]
df_correlations = df_correlations[df_correlations['previous_star_ratings'] == df_correlations['previous_star_ratings']]
print(df_correlations)
print(df_correlations.corr(method='pearson'))
print(df_correlations.corr(method='pearson')['star_rating']['previous_star_ratings'])

df_correlations.plot.scatter(x='all_previous_star_ratings', y='previous_star_ratings')
# z = np.polyfit(df_correlations['all_previous_star_ratings'], df_correlations['star_rating'], 1)
# p = np.poly1d(z)
# plt.plot(df_correlations['all_previous_star_ratings'],p(df_correlations['all_previous_star_ratings']),"r--")
df_correlations[['star_ratings', 'previous_star_ratings']].plot()
plt.show()

