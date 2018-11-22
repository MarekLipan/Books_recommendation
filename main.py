"""
MAIN SCRIPT

This script is used to load and manipulate the data.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# paths
path = "/Users/marek/Desktop/Work/Book_recommendation/"
data_path = path + "Data/"
figure_path = path + "Presentation/"

########################
# BOOK-CROSSING DATASET#
########################

# book-crossing dataset
bc_ratings = pd.read_csv(data_path + "BX-CSV-Dump/" + "BX-Book-Ratings.csv",
                         delimiter=";", encoding="latin-1")
bc_books = pd.read_csv(data_path + "BX-CSV-Dump/" + "BX-Books.csv",
                       delimiter=";", encoding="latin-1",
                       error_bad_lines=False)

# drop columns unimportant for our analysis
bc_books = bc_books.iloc[:, :4]

# keep only books with ISBN of correct format
bc_ratings["ISBN"] = bc_ratings["ISBN"].str.replace("X$", "") # delete digit check X
bc_ratings = bc_ratings[~bc_ratings["ISBN"].str.match("\d*\D+\d*")]
bc_books["ISBN"] = bc_books["ISBN"].str.replace("X$","") # delete digit check X
bc_books = bc_books[~bc_books["ISBN"].str.match("\d*\D+\d*")]

# find out the ISBNs of the Lord of the Rings books
bc_LotR_ISBN = bc_books[bc_books["Book-Title"].str.match(".*The Lord of the Rings.*")]
# The Fellowship of the Ring (The Lord of the Rings, Part 1)
# The Two Towers (The Lord of the Rings, Part 2)
# The Return of the King (The Lord of the Rings, Part 3)
bc_LotR_ISBN = pd.concat([
        bc_books[bc_books["Book-Title"].str.upper() == str.upper("The Fellowship of the Ring (The Lord of the Rings, Part 1)")],
        bc_books[bc_books["Book-Title"].str.upper() == str.upper("The Two Towers (The Lord of the Rings, Part 2)")],
        bc_books[bc_books["Book-Title"].str.upper() == str.upper("The Return of the King (The Lord of the Rings, Part 3)")]
        ])["ISBN"]

# group the LotR books into one (ISBN: 0345339703) with average rating per user
LotR_ISBN = "0345339703"
bc_ratings["ISBN"].replace(bc_LotR_ISBN.values, LotR_ISBN, inplace=True)
bc_ratings = bc_ratings.groupby(["User-ID", "ISBN"])["Book-Rating"].mean().to_frame()
bc_ratings.reset_index(inplace=True)

# group books with the same name under one ISBN (different editions)
bc_ratings = bc_ratings.merge(bc_books.iloc[:,:2], left_on="ISBN", right_on='ISBN', how='left').dropna()
# I only care about unique titles and not authors, because more damage is caused by different author spelling than common book names
bc_books.drop_duplicates(subset="Book-Title", keep="first", inplace=True)
bc_ratings = bc_ratings.merge(bc_books.iloc[:,:3], left_on="Book-Title", right_on='Book-Title', how='left')
bc_ratings.drop("ISBN_x", axis=1, inplace=True)
bc_ratings.rename(columns={"ISBN_y": "ISBN"}, inplace=True)
bc_ratings = bc_ratings.groupby(["User-ID", "ISBN","Book-Title"])["Book-Rating"].mean().to_frame()
bc_ratings.reset_index(inplace=True)

# filter out users with too low amount of ratings given (reliability reasons)
rel_users = bc_ratings["User-ID"].value_counts()
rel_users = rel_users[rel_users >= 5].index.values
bc_ratings = bc_ratings[bc_ratings["User-ID"].isin(rel_users)]

# filter out book with too low amount of ratings obtained (reliability reasons)
rel_books = bc_ratings["ISBN"].value_counts()
rel_books = rel_books[rel_books >= 20].index.values
bc_ratings = bc_ratings[bc_ratings["ISBN"].isin(rel_books)]

# len(bc_ratings["User-ID"].unique())
# number of users: 18982

# len(bc_ratings["ISBN"].unique())
# number of books: 6724

# max(bc_ratings["Book-Rating"])
# min(bc_ratings["Book-Rating"])
# implicit book ratings range: 0-10

# histogram of LotR ratings
fig, a = plt.subplots(figsize=(8, 6))
a.hist(bc_ratings.loc[bc_ratings["ISBN"] == LotR_ISBN,"Book-Rating"], color="b", density=False)
a.set_xlabel('Rating')
a.set_ylabel('Count')
a.set_title("LotR Ratings")
a.grid(color='k', linestyle=':', linewidth=0.5)
fig.savefig(figure_path + "LotR_ratings_hist.pdf", bbox_inches='tight')

# find users who are fans of the LotR books (on average >8 vote)
LotR_ratings = bc_ratings[bc_ratings["ISBN"] == LotR_ISBN]
LotR_fans = LotR_ratings.loc[LotR_ratings["Book-Rating"] > 8, "User-ID"].unique()

##############
# END OF FILE#
##############
