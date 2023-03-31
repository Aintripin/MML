import pandas as pd
import matplotlib.pyplot as plt

# Загрузка датасета
data = pd.read_csv('titanic.csv')

# Заполнение пропущенных значений медианным возрастом:
median_age = data['Age'].median()
data['Age'].fillna(median_age, inplace=True)

# Выбор столбца Age
age = data["Age"]

from scipy.stats import iqr

# Вычисление межквартильного размаха
q1, q3 = age.quantile(0.25), age.quantile(0.75)
iqr_age = iqr(age)

# Определение границ выбросов
outlier_min = q1 - 1.5 * iqr_age
outlier_max = q3 + 1.5 * iqr_age

# Удаление выбросов
age_clean = age[(age >= outlier_min) & (age <= outlier_max)]

print("Межквартильный размах: ", iqr_age)
print("Нижняя граница выбросов: ", outlier_min)
print("Верхняя граница выбросов: ", outlier_max)

# Построение гистограммы исходного признака
plt.hist(age, bins=20)
plt.title("Исходный признак Age")
plt.xlabel("Age")
plt.ylabel("Частота")
plt.show()

# Построение гистограммы очищенного от выбросов признака
plt.hist(age_clean, bins=20)
plt.title("Очищенный от выбросов признак Age")
plt.xlabel("Age")
plt.ylabel("Частота")
plt.show()
