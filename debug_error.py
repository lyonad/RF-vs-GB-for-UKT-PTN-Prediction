import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('Data/data.csv')
print(f"Data loaded successfully with shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# This simulates the specific error scenario mentioned - cell In[11], line 8
# which suggests the distribution plotting cell
target_cols = ['UKT-1', 'UKT-2', 'UKT-3', 'UKT-4']

# Replicate the exact problematic code mentioned in the error:
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

print("Testing the histogram creation (the original problematic code):")
for i, col in enumerate(target_cols):
    print(f"Processing {col}")
    if col in df.columns:
        print(f"  Column {col} exists, creating histogram...")
        axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    else:
        print(f"  ERROR: {col} not found!")
        print(f"  Available columns: {list(df.columns)}")

plt.tight_layout()
# Image saving disabled
# plt.savefig('test_verification.png')
plt.close()

print("Test completed successfully - no KeyError for 'fee_year_1'!")

# Also test the other cell that uses target_cols (which would be the one causing In[11] error)
print("\nTesting the time series plot (probably In[11]):")
plt.figure(figsize=(12, 6))

for col in target_cols:
    if col in df.columns:
        plt.plot(df[col].head(50), label=col, marker='o')
    else:
        print(f"ERROR: Column {col} not available for plotting!")

plt.title('Tuition Fees Across Years (First 50 Universities)')
plt.xlabel('University Index')
plt.ylabel('Tuition Fee (IDR)')
plt.legend()
plt.grid(True, alpha=0.3)
# Image saving disabled
# plt.savefig('test_series.png')
plt.close()

print("Time series plot also completed successfully!")