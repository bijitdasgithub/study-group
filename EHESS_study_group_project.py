# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:37:39 2023

@author: bijit
"""

import nltk
ntlk.download('stopwords')
import sqlite3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Function to generate keywords from summary
def generate_keywords(summary):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(summary)
    keywords = [word for word in words if word.lower() not in stop_words]
    return keywords

# Connect to database
conn = sqlite3.connect("thesis_database.db")

# Create table to store thesis data
conn.execute('''CREATE TABLE IF NOT EXISTS theses
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
             name TEXT NOT NULL,
             email TEXT NOT NULL,
             course TEXT NOT NULL,
             year INTEGER NOT NULL,
             summary TEXT NOT NULL,
             keywords TEXT NOT NULL)''')

# Prompt user to enter thesis data
name = input("Enter your name: ")
email = input("Enter your email id: ")
course = input("Enter your course name: ")
year = int(input("Enter your year: "))
summary = input("Enter summary of your thesis: ")
keywords = generate_keywords(summary)
print("Keywords generated from summary: {}".format(", ".join(keywords)))

# Prompt user to modify keywords if desired
while True:
    modify_keywords = input("Do you want to modify the keywords? (y/n) ")
    if modify_keywords.lower() == "n":
        break
    elif modify_keywords.lower() == "y":
        action = input("Do you want to add, remove, or keep the same keywords? ")
        if action.lower() == "add":
            new_keywords = input("Enter new keywords separated by commas: ")
            new_keywords_list = [keyword.strip() for keyword in new_keywords.split(",")]
            keywords += new_keywords_list
        elif action.lower() == "remove":
            remove_keywords = input("Enter keywords to remove separated by commas: ")
            remove_keywords_list = [keyword.strip() for keyword in remove_keywords.split(",")]
            keywords = [keyword for keyword in keywords if keyword not in remove_keywords_list]
        print("Modified keywords: {}".format(", ".join(keywords)))

# Store thesis data in database
conn.execute("INSERT INTO theses (name, email, course, year, summary, keywords) VALUES (?, ?, ?, ?, ?, ?)",
             (name, email, course, year, summary, ", ".join(keywords)))
conn.commit()

# Use TF-IDF vectorizer to transform keywords into a feature matrix
cursor = conn.execute("SELECT id, keywords FROM theses")
papers = [{"id": row[0], "keywords": row[1].split(", ")} for row in cursor]
vectorizer = TfidfVectorizer()
keywords_matrix = vectorizer.fit_transform([", ".join(paper["keywords"]) for paper in papers])

# Use K-means clustering to group similar theses together
num_clusters = int(input("Enter number of clusters to create: "))
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(keywords_matrix)

# Print out the theses in each cluster
for i in range(num_clusters):
    cluster_indices = [j for j, label in enumerate(kmeans.labels_) if label == i]
    cluster_theses = [(papers[j]["id"], papers[j]["keywords"]) for j in cluster_indices]
    print("Cluster {}: {}".format(i+1, ", ".join([str(th[0]) for th in cluster_theses])))
