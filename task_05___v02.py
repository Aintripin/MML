import pandas as pd

# загрузка набора данных
data = pd.read_csv("titanic.csv")

# выбор признака для кодирования
feature_to_encode = "Sex"

# кодирование методом "one-hot encoding"
one_hot_encoded = pd.get_dummies(data[feature_to_encode], prefix=feature_to_encode)

# добавление закодированных признаков в набор данных
data = pd.concat([data, one_hot_encoded], axis=1)

# удаление исходного признака
data = data.drop(columns=[feature_to_encode])

# экспорт результатов в файл Excel
data.to_excel("titanic_encoded.xlsx", index=False)
