import pandas as pd
import numpy as np

np.random.seed(43)

# Размер генерируемого датасурса
num_samples = 100000

age = np.random.randint(18, 65, num_samples)
sex = np.random.choice([0, 1], num_samples)
num_purchases = np.random.randint(0, 100, num_samples)
calls = np.random.randint(1, 15, num_samples)
churn = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])

data = pd.DataFrame({
    'Age': age,
    'Sex': sex,
    'NumPurchases': num_purchases,
    'Calls': calls,
    'Churn': churn
})

data.to_csv('data.csv', index=False)



