Описание данных
Исторические данные о температуре содержатся в файле temperature_data.csv, включают:

city: Название города.
timestamp: Дата (с шагом в 1 день).
temperature: Среднесуточная температура (в °C).
season: Сезон года (зима, весна, лето, осень).
Код для генерации файла вы найдете ниже.

Этапы выполнения
Анализ исторических данных:

Вычислить скользящее среднее температуры с окном в 30 дней для сглаживания краткосрочных колебаний.
Рассчитать среднюю температуру и стандартное отклонение для каждого сезона в каждом городе.
Выявить аномалии, где температура выходит за пределы  среднее±2𝜎 .
Попробуйте распараллелить проведение этого анализа. Сравните скорость выполнения анализа с распараллеливанием и без него.
Мониторинг текущей температуры:

Подключить OpenWeatherMap API для получения текущей температуры города. Для получения API Key (бесплатно) надо зарегистрироваться на сайте. Обратите внимание, что API Key может активироваться только через 2-3 часа, это нормально. Посему получите ключ заранее.
Получить текущую температуру для выбранного города через OpenWeatherMap API.
Определить, является ли текущая температура нормальной, исходя из исторических данных для текущего сезона.
Данные на самом деле не совсем реальные (сюрпрайз). Поэтому на момент эксперимента погода в Берлине, Каире и Дубае была в рамках нормы, а в Пекине и Москве аномальная. Протестируйте свое решение для разных городов.
Попробуйте для получения текущей температуры использовать синхронные и асинхронные методы. Что здесь лучше использовать?
Создание приложения на Streamlit:

Добавить интерфейс для загрузки файла с историческими данными.
Добавить интерфейс для выбора города (из выпадающего списка).
Добавить форму для ввода API-ключа OpenWeatherMap. Когда он не введен, данные для текущей погоды не показываются. Если ключ некорректный, выведите на экран ошибку (должно приходить {"cod":401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."}).
Отобразить:
Описательную статистику по историческим данным для города, можно добавить визуализации.
Временной ряд температур с выделением аномалий (например, точками другого цвета).
Сезонные профили с указанием среднего и стандартного отклонения.
Вывести текущую температуру через API и указать, нормальна ли она для сезона.

Дополнительно добавила карту на которой есть точка этого города.
