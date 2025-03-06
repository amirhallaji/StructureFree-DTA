
from datasets import load_dataset
dataset = load_dataset("amirhallaji/davis")['train'].to_pandas()

print(dataset)
# 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

from sklearn.model_selection import train_test_split
# split dataset with random state 0 into train and val and save them in data/
train, val = train_test_split(dataset, test_size=0.2, random_state=0)
train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)

