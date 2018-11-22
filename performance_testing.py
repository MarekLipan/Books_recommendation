"""
PERFORMANCE TESTING

"""

import pandas as pd
import numpy as np
import recommendation_methods as rm

def cross_val(alpha, ratings, books, fan_group, target_ISBN):
    """
    The 4-fold cross-validation function, which determines average users Rating
    per list of top 10 recommended books.

    Parameters
    ----------
    alpha : Float
        The combining weight from [0,1] in the following formula:

            recom_rank = alpha * pop_rank + (1-alpha) * sim_rank

    ratings : DataFrame
        DataFrame of ratings containing User-ID, ISBN, Book-Title, Book-Rating.

    books : DataFrame
        DataFrame of ratings containing ISBN, Book-Title, Book-Author.

    fan_group : Array
        Numpy Array containing User-IDs of the fans of the book of our interest.

    target_ISBN : Str
        String containing ISBN of the book of our interest.

    Returns
    -------
    list_rating : Float
        The average fans average rating assigned to the recommended list.
    """

    # randomly assign users to cross-validation folds
    fold_size = np.floor(fan_group.shape[0]/4)
    fold = np.concatenate([
            np.repeat(0, fold_size),
            np.repeat(1, fold_size),
            np.repeat(2, fold_size),
            np.repeat(3, fan_group.shape[0] - 3*fold_size)
            ])
    fold = np.random.permutation(fold)

    # initialize cross-validation list_rating
    list_rating = 0

    # in each of the 4 steps use all but the particular folds fan group users
    for k in range(4):
        test_fans = fan_group[fold == k]
        train_fans = fan_group[fold != k]
        test_ratings = ratings[ratings["User-ID"].isin(test_fans)]
        train_ratings = ratings[~ratings["User-ID"].isin(test_fans)]

        # create the top 10 recommendation list
        rec_list = rm.hybrid(0.8, train_ratings, books, train_fans, target_ISBN)[:10][["Book-Title"]]

        # obtain the user average ratings
        list_rating += test_ratings[test_ratings["Book-Title"].isin(rec_list["Book-Title"])].groupby(["User-ID"])["Book-Rating"].mean().mean()/4

    return list_rating

# run cross-validation with different weights (alpha)
CV_alpha = pd.DataFrame({"alpha" : np.linspace(0, 1, num=20, endpoint=True, dtype=float),
                         "list_rating" : np.full(20, 0, dtype=float)})
np.random.seed(444) # for replicability of results
for a in range(20):
    CV_alpha.iloc[a, 1] = cross_val(CV_alpha.iloc[a, 0], bc_ratings, bc_books, LotR_fans, LotR_ISBN)

alpha_star = CV_alpha.iloc[CV_alpha["list_rating"].values.argmax(), 0]

# cross-validation plot
fig, a = plt.subplots(figsize=(8, 6))
plt.plot(CV_alpha["alpha"], CV_alpha["list_rating"], marker='o', linewidth=1, color="b")
a.set_title('Avg. rating of recommended books: 4-fold cross-validation')
a.set_xlabel('Alpha')
a.set_ylabel('Rating')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.axvline(x=alpha_star, color="red")
fig.savefig(figure_path + "cv_figure.pdf", bbox_inches='tight')

# run the methods on the whole sample (obtain top 10 rec. list)
pop_recom = rm.fan_pop(bc_ratings, bc_books, LotR_fans, LotR_ISBN)[:10][["Book-Title", "Book-Author"]]
sim_recom = rm.book_simil(bc_ratings, bc_books, LotR_fans, LotR_ISBN)[:10][["Book-Title", "Book-Author"]]
hybrid_recom = rm.hybrid(alpha_star, bc_ratings, bc_books, LotR_fans, LotR_ISBN)[:10][["Book-Title", "Book-Author"]]

# hybrid method with diversifications heuristic: only one recommendation per
# autor limit in the top 10 list
hybrid_recom_div = rm.hybrid(alpha_star, bc_ratings, bc_books, LotR_fans, LotR_ISBN)
hybrid_recom_div["Author"] = hybrid_recom_div["Book-Author"].str.replace(" ","").str.upper()
hybrid_recom_div = hybrid_recom_div.drop_duplicates(subset="Author", keep="first")[:10][["Book-Title", "Book-Author"]]

##############
# END OF FILE#
##############
