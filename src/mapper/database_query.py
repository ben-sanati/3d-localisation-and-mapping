import pandas as pd


class PoseDataExtractor:
    def __init__(self, pose_path):
        self.pose_path = pose_path 

    def fetch_data(self):
        df = pd.read_csv(self.pose_path, sep=' ', skiprows=1, header=None)
        df.columns = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df


if __name__ == '__main__':
    pose_path = '../common/data/gold_std/poses.txt'
    extractor = PoseDataExtractor(pose_path)
    data = extractor.fetch_data()
    print(data)
