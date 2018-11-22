"""
RECOMMENDATION METHODS SCRIPT

"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

########################
# Fan Popularity method#
########################

def fan_pop(ratings, books, fan_group, target_ISBN):
    """
    This recommends books according to their average rating in the fan group.

    Parameters
    ----------
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
    recom : DataFrame
        The ordered DataFrame of recommended books: Book-title, Book-author.
        The rows are order from the most recommended book in the top row to the
        least recommended in the bottom row.
    """

    # most popular books in the fan group
    pop_books = ratings.loc[ratings["User-ID"].isin(fan_group), ["ISBN", "Book-Rating"]]
    pop_books = pop_books.groupby(["ISBN"])["Book-Rating"].agg(['mean', 'count'])

    # consider only book with at least several ratings in the fan group
    pop_books = pop_books[pop_books["count"] >= 5]

    # other books than those (commonly) rated by the fan group are assigned rating 0
    pop_books = pd.DataFrame(ratings["ISBN"].unique(), columns=["ISBN"]).merge(
            pop_books, left_on='ISBN', right_index=True, how='left')
    pop_books.fillna(0, inplace=True)

    # recommend other than the LotR books
    pop_books = pop_books[pop_books["ISBN"] != target_ISBN]

    # recommended the books based on the average rating
    recom = pop_books.merge(books, left_on='ISBN', right_on='ISBN', how='left')
    recom.rename(columns={"mean": "Book-Rating"}, inplace=True)
    recom = recom.loc[:, ["Book-Rating", "Book-Title", "Book-Author"]].sort_values(["Book-Rating", "Book-Title"], ascending=False)

    return recom

# run on the whole sample
#pop_recom = fan_pop(bc_ratings, bc_books, LotR_fans, LotR_ISBN)

#########################
# Book similarity method#
#########################

def book_simil(ratings, books, fan_group, target_ISBN):
    """
    This recommends books based the similarity of the target book and its cosine
    similarity to other books.

    Parameters
    ----------
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
    recom : DataFrame
        The ordered DataFrame of recommended books: Book-title, Book-author.
        The rows are order from the most recommended book in the top row to the
        least recommended in the bottom row.
    """

    # create Users X ISBN matrix
    rat_mat = ratings.pivot(index="User-ID", columns="ISBN", values="Book-Rating")
    rat_mat.fillna(0, inplace=True)
    book_sim = cosine_similarity(
            X=rat_mat.transpose(),
            Y=rat_mat[target_ISBN].to_frame().transpose()
            )
    book_sim = pd.DataFrame(data=book_sim, index=rat_mat.columns, columns=["similarity"])

    # recommend books based on similarity
    recom = book_sim.merge(books, left_index=True, right_on='ISBN', how='left').sort_values(["similarity", "Book-Title"], ascending=False)
    recom = recom[1:]  # remove the book itself (similarity=1)
    recom = recom.loc[:, ["similarity", "Book-Title", "Book-Author"]]

    return recom

# run on the whole sample
#sim_recom = book_simil(bc_ratings, bc_books, LotR_fans, LotR_ISBN)

####################
# The hybrid method#
####################

def hybrid(alpha, ratings, books, fan_group, target_ISBN):
    """
    The method recommends books based on a linear combination of recommendation
    ranks from the popularity and similarity methods.

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
    recom : DataFrame
        The ordered DataFrame of recommended books: Book-title, Book-author.
        The rows are order from the most recommended book in the top row to the
        least recommended in the bottom row.
    """
    # obtain recommendations from the popularity method
    pop_recom = fan_pop(ratings, books, fan_group, target_ISBN)
    pop_recom["rank"] = range(1, pop_recom.shape[0]+1)

    # obtain recommendations from the similarity method
    sim_recom = book_simil(ratings, books, fan_group, target_ISBN)
    sim_recom["rank"] = range(1, sim_recom.shape[0]+1)

    # linearly combine the recommendations
    recom = pop_recom.merge(sim_recom[["Book-Title", "rank"]], left_on="Book-Title", right_on="Book-Title", how='left')
    recom["rank"] = alpha*recom["rank_x"] + (1-alpha)*recom["rank_y"]
    recom.sort_values(["rank", "Book-Title"], ascending=True, inplace=True)

    return recom


# run on the whole sample
#hybrid_recom = hybrid(0.8, bc_ratings, bc_books, LotR_fans, LotR_ISBN)




##############
# END OF FILE#
##############
