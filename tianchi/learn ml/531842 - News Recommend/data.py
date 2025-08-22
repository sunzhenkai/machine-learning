import os

class DataManager:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.article_csv_path = f'{base_path}/articles.csv'
        self.article_emb_csv_path = f'{base_path}/articles_emb.csv'
        self.train_click_csv_path = f'{base_path}/train_click_log.csv'
        self.test_click_csv_path = f'{base_path}/testA_click_log.csv'

    def describe():
        pass


if __name__ == '__main__':
    dm = DataManager()
    print('run main done')