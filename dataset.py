from torch.utils.data import Dataset
from tqdm import tqdm

from settings import cursor
from utils import train_test_split


class MimicDataset(Dataset):
    def __init__(self, query, batch_size=32) -> None:
        super().__init__()
        self.query = query
        self.batch_size = batch_size

        cursor.execute("""
            SELECT count(*)
            FROM mimiciii.diagnoses_icd d JOIN mimiciii.noteevents n
            ON d.subject_id = n.subject_id
        """)
        self.total_size = cursor.fetchone()[0]

    def __len__(self):
        return self.total_size // self.batch_size

    def __getitem__(self, index):
        start_idx = index * self.batch_size

        cursor.execute(self.query + f" OFFSET {start_idx} LIMIT {self.batch_size}")
        rows = cursor.fetchall()

        return rows[index][0], rows[index][1]


query = """
SELECT n.text, d.icd9_code
FROM mimiciii.diagnoses_icd d JOIN mimiciii.noteevents n
ON d.subject_id = n.subject_id
"""

mimic_dataset = MimicDataset(query)
train_loader, test_loader = train_test_split(mimic_dataset)

if __name__ == '__main__':
    print(len(mimic_dataset))
    for text, code in train_loader:
        print(text[:20] + "...", code)
