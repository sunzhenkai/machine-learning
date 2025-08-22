import os
import pandas as pd

class DataManager:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.article_csv_path = f'{base_path}/articles.csv'
        self.article_emb_csv_path = f'{base_path}/articles_emb.csv'
        self.train_click_csv_path = f'{base_path}/train_click_log.csv'
        self.test_click_csv_path = f'{base_path}/testA_click_log.csv'
        # pandas
        self.df_article = pd.read_csv(self.article_csv_path)
        self.df_article_emb = pd.read_csv(self.article_emb_csv_path)
        self.df_train_click = pd.read_csv(self.train_click_csv_path)
        self.df_test_click = pd.read_csv(self.test_click_csv_path)

    def describe(self):
        print('----- article -----')
        print(self.df_article.describe())
        print('\n----- article emb -----')
        print(self.df_article_emb.describe())
        print('\n----- train click -----')
        print(self.df_train_click.describe())
        print('\n----- test click -----')
        print(self.df_test_click.describe())


if __name__ == '__main__':
    dm = DataManager('~/dataset/tianchi-news-rec')
    dm.describe()
    print('run main done')