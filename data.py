import matplotlib.pyplot as plt
import pandas as pd

#add file
file_path = r'D:\Neeraj\UoN\Machine Learning\Assignment 2\TrainDataset2025 (1).csv'


df = pd.read_csv(file_path, header=0)
df.rename(columns={df.columns[0]: 'ID', df.columns[1]: 'pCR', df.columns[2]:'RFS'}, inplace=True)

#add any feature you want
target = 'Gene'
counts_initial = df[target].value_counts()

plt.figure(figsize=(8, 6))
bars_initial = plt.bar(
    counts_initial.index.astype(str),
    counts_initial.values,
    color=['green', 'red', 'blue', 'orange'] 
)

for bar in bars_initial:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 5,
        f'{height}',
        ha='center',
        fontsize=12,
        fontweight='bold'
    )

plt.xlabel('Value')
plt.ylabel('Count')
plt.title('Fig1 Distribution (Before replacing 999 with 1)')
plt.show()


df.replace([999, 999.0], 1, inplace=True)


counts_final = df[target].value_counts()


plt.figure(figsize=(8, 6))
bars_final = plt.bar(
    counts_final.index.astype(str),
    counts_final.values,
    color=['green', 'red'] 
)


for bar in bars_final:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 5,
        f'{height}',
        ha='center',
        fontsize=12,
        fontweight='bold'
    )

plt.xlabel('Value')
plt.ylabel('Count')
plt.title('Figure 2: Distribution of Selected Value (After replacing 999 with 1)')
plt.show()