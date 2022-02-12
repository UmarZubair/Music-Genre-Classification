
class GTZANDataset:
    def __init__(self, df):
        self.x_paths = df['paths']
        self.labels = df['target']

    def __getitem__(self, idx):
        x = lb.load(x_paths[idx])
        y = labels[idx]