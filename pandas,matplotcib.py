import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Year': [2018, 2019, 2020, 2021, 2022],
    'Sales': [150, 200, 170, 240, 300]
}

df = pd.DataFrame(data)

print("Dataset:\n", df)
print("\nBasic Statistics:\n", df.describe())

plt.plot(df['Year'], df['Sales'], marker='o', color='blue')
plt.title("Yearly Sales")
plt.xlabel("Year")
plt.ylabel("Sales")
plt.grid(True)
plt.show()