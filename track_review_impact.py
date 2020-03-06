import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# TODO
# Instead of looking at the average star rating per day, look at individual reviews
# When looking at past reviews, take the weights of the reviews into account

min_review_num = 5

#microwave, pacifier, hair_dryer
df = pd.read_csv('microwave.tsv', sep='\t')

names_of_products = df['product_title'].tolist()

dict_names_count = Counter(names_of_products)
dict_names_count_filtered = dict()
names_of_products_filtered = []
for (key, value) in dict_names_count.items():
    if(value > min_review_num):
        names_of_products_filtered.append(key)

correlations = []
num_reviews = []

for name in names_of_products_filtered:
    print(name)
    df_product = df[df['product_title'] == name]
    df_product_filtered  = df_product[df_product['helpful_votes'] > 0]
    df_grouped_product = df_product_filtered.groupby('review_date').mean().sort_values(by=['review_date']) # maybe add in weights for upvotes later
    previous_star_reviews = df_grouped_product.star_rating.shift(1) + df_grouped_product.star_rating.shift(2) + df_grouped_product.star_rating.shift(3)
    previous_star_reviews = previous_star_reviews.rename(columns={"stars":"previous_stars"})
    df_grouped_product['previous_stars'] = previous_star_reviews
    df_grouped_product['previous_stars'] = df_grouped_product['previous_stars'] / 3
    correlations.append(df_grouped_product.corr(method='pearson')['star_rating']['previous_stars'])
    num_reviews.append(len(df_product.index))

correlations = [x for x in correlations if str(x) != 'nan']
print(num_reviews)

n, bins, patches = plt.hist(correlations)
plt.show()
