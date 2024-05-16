import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def load_data():
    uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, sep=';')
        return data
    return None

def main():
    st.title("Интерактивный дашборд для анализа данных")
    
    data = load_data()

    if data is not None:
        st.write("Предварительный просмотр данных:", data.head())

        # Выбор целевой переменной
        target = st.selectbox("Выберите целевую переменную", data.columns)

        # Выбор метода понижения размерности
        dimension_reduction = st.selectbox("Выберите метод понижения размерности", ["PCA", "TruncatedSVD", "Нет"])

        # Подготовка данных
        X = data.drop(target, axis=1)
        y = data[target]
        st.write("Характеристики прогнозируемого столбца:", y.describe())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if dimension_reduction == "PCA":
            pca = PCA(n_components=0.95)  # Сохраняем 95% дисперсии
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        elif dimension_reduction == "TruncatedSVD":
            svd = TruncatedSVD(n_components=50)  # Примерное количество компонент
            X_train = svd.fit_transform(X_train)
            X_test = svd.transform(X_test)

        optimize = st.selectbox("Оптимизировать гиперпараметры?", ["Да", "Нет"])

        # Кнопка для запуска обучения модели
        if st.button("Запустить обучение модели"):
            if optimize == "Да":
                # Начало прогресс-бара
                my_bar = st.progress(0)
                clf = RandomForestRegressor()
                param_grid = {
                    'n_estimators': [100, 200, 300, 400, 500],
                    'max_depth': [10, 30, 50, 70, 100],
                    'max_features': ['auto', 'sqrt'],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                }

                # RandomizedSearch
                rs = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=10, cv=3, random_state=42, n_jobs=-1)
                rs.fit(X_train, y_train)
                my_bar.progress(50)  # Обновление прогресс-бара
                best_params = rs.best_params_

                # GridSearch в окрестности найденных параметров
                param_grid = {
                    'n_estimators': [best_params['n_estimators'] - 50, best_params['n_estimators'], best_params['n_estimators'] + 50],
                    'max_depth': [best_params['max_depth'] - 10, best_params['max_depth'], best_params['max_depth'] + 10],
                    'max_features': [best_params['max_features']],
                    'min_samples_split': [best_params['min_samples_split'] - 2, best_params['min_samples_split'], best_params['min_samples_split'] + 2],
                    'min_samples_leaf': [best_params['min_samples_leaf'] - 1, best_params['min_samples_leaf'], best_params['min_samples_leaf'] + 1],
                    'bootstrap': [best_params['bootstrap']]
                }
                gs = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                my_bar.progress(100)  # Завершение прогресс-бара

                # Обучение модели с оптимизированными гиперпараметрами
                best_clf = gs.best_estimator_
                best_clf.fit(X_train, y_train)
                predictions = best_clf.predict(X_test)

                # Вывод результатов
                st.write("Лучшие параметры:", gs.best_params_)
                st.write("Mean Squared Error:", mean_squared_error(y_test, predictions))
                st.write("R-squared:", r2_score(y_test, predictions))

            else:
                # Обучение модели с параметрами по умолчанию
                clf = RandomForestRegressor()
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)

                # Вывод результатов
                st.write("Mean Squared Error:", mean_squared_error(y_test, predictions))
                st.write("R-squared:", r2_score(y_test, predictions))

if __name__ == "__main__":
    main()

#streamlit run dashboard.py