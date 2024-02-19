import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Загрузка набора данных Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Вычисление статистических данных
median = df.median()
correlation = df.corr()

# Функция для вычисления среднего значения
def calc_mean(data):
    return sum(data) / len(data)

# Функция для вычисления моды
def calc_mode(data):
    counts = {value: data.count(value) for value in data}
    max_count = max(counts.values())
    modes = [value for value, count in counts.items() if count == max_count]
    return modes

# Функция для вычисления стандартного отклонения
def calc_std_dev(data):
    mean = calc_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return variance ** 0.5

# Развернутый расчет корреляции
def calculate_correlation_matrix(data_list):
    n = len(data_list[0])
    corr_matrix = [[0] * n for _ in range(n)]
    means = [sum(row[i] for row in data_list) / len(data_list) for i in range(n)]

    for i in range(n):
        for j in range(i, n):
            numerator = sum((row[i] - means[i]) * (row[j] - means[j]) for row in data_list)
            denominator = (sum((row[i] - means[i]) ** 2 for row in data_list) ** 0.5) * (
                        sum((row[j] - means[j]) ** 2 for row in data_list) ** 0.5)

            corr = numerator / denominator if denominator != 0 else 0
            corr_matrix[i][j] = corr
            corr_matrix[j][i] = corr

    return corr_matrix


# Преобразование данных в формат списка списков
data_list = df.values.tolist()

# Вычисление матрицы корреляции
correlation_matrix = calculate_correlation_matrix(data_list)

# Отображение результатов
print("Correlation2:")
for row in correlation_matrix:
    print(row)

# Функция для вычисления медианы
def calc_median(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    if n % 2 == 0:
        median1 = sorted_data[n//2]
        median2 = sorted_data[n//2 - 1]
        median3 = (median1 + median2)/2
    else:
        median = sorted_data[n//2]
    return median3

# Вычисление медианы для каждого столбца с использованием функции calc_median
custom_median = {column: calc_median(df[column]) for column in df.columns}

print(f"Median2:\n{custom_median}\n")

std_dev = df.std()
mode = df.mode()
mean = df.mean()

custom_mean = {column: calc_mean(df[column].tolist()) for column in df.columns}
custom_mode = {column: calc_mode(df[column].tolist()) for column in df.columns}
custom_std_dev = {column: calc_std_dev(df[column].tolist()) for column in df.columns}

# Вывод статистических данных
print(f"Mean2:\n{custom_mean}\n")
print(f"Mode2:\n{custom_mode}\n")
print(f"Standard Deviation2:\n{custom_std_dev}\n")
print(f"Median:\n{median}\n")
print(f"Correlation:\n{correlation}\n")
print(f"Standard Deviation:\n{std_dev}\n")
print(f"Mode:\n{mode}\n")
print(f"Mean:\n{mean}\n")

# Создание графика
sns.pairplot(df)
plt.show()
