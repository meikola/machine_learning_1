import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных из Excel-файла
file_path = r"C:\Users\Admin\Desktop\1.xlsx"
df = pd.read_excel(file_path)

# Предварительная обработка данных
# Пример: Кодирование категориальных значений
df['Experience'] = pd.Categorical(df['Experience'])
df['Experience_code'] = df['Experience'].cat.codes

# Создание нового столбца 'Salary_Class' в соответствии с условиями
conditions = [
    (df['Annual Salary'] > 500000),
    (df['Annual Salary'] <= 500000) & (df['Annual Salary'] > 300000),
    (df['Annual Salary'] <= 300000) & (df['Annual Salary'] > 100000),
    (df['Annual Salary'] <= 100000)
]
values = ['Super High Salary', 'High Salary', 'Medium Salary', 'Low Salary']

df['Salary_Class'] = np.select(conditions, values)

# Удаление строк с NaN
df = df.dropna(subset=['Experience_code', 'Annual Salary', 'Salary_Class'])

# Сортировка данных
df = df.sort_values(by=['Experience_code', 'Annual Salary'])

# Создание графика для каждого класса зарплат
sns.set(style="whitegrid")
plt.figure(figsize=(12, 10))
sns.scatterplot(x='Experience_code', y='Annual Salary', hue='Salary_Class', data=df, palette=['purple', 'red', 'orange', 'green'], s=100)
plt.title('Experience vs. Annual Salary (Color by Salary Class)')
plt.xlabel('Experience')
plt.ylabel('Annual Salary')
plt.show()
