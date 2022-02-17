from utils import * 
from feature_extract import extract_mel_band_energies
import matplotlib.pyplot as plt
import pandas as pd

def main():
    file_path = 'data/blues/blues.00000.au'
    audio, sr = get_audio_file_data(file_path)
    print(sr)
    print(np.shape(audio))
    f = extract_mel_band_energies(audio)
    print(np.shape(f))
    plt.imshow(f)
    plt.show()
    
def test():
    test_csv = pd.read_csv('data/test_metadata.csv')
    train_csv = pd.read_csv('data/train_metadata.csv')
    print(test_csv.head())
    print(train_csv.head())
    
if __name__ == "__main__":
    #main()
    test()