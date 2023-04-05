import os
import json
import pandas as pd




def list2csv(labels, articles):
    df = pd.DataFrame()
    df['features'] = articles
    df['labels'] = labels
    print(df)
    df.to_csv('articles_labels.csv', index=False)



def load_data():
    root_labels = "./Basil-main/annotations"
    root_articles = "./Basil-main/articles"

    data = {}
    labels = []
    articles = []

    for year in os.listdir(root_labels):
        year_d_labels = root_labels + "/" + str(year)
        year_d_articles = root_articles + "/" + str(year)
        temp_dict = {}
        if year.isnumeric():
            for article_label in os.listdir(year_d_labels):
                # Load Labels
                labels_file = open(year_d_labels + "/" + article_label)
                json_labels = json.load(labels_file)
                labels.append(json_labels["article-level-annotations"]["relative_stance"])
            for article_article in os.listdir(year_d_articles):
                # Load Labels
                article_file = open(year_d_articles + "/" + article_article)
                json_article = json.load(article_file)

                article_str = ""
                for sentance in json_article["body-paragraphs"]:
                    article_str += sentance[0]
                articles.append(article_str)
    return labels, articles


def run_dataLoader():
    labels, articles = load_data()
    list2csv(labels, articles)




run_dataLoader()