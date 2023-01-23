# Классификация музыкальных жанров из базе эмбеддингов music2vec-v1[1]

Данная лабараторная работа заключалась в построении классификатора музыкальных жанров на базе готовых аудио-эмбеддингов.

В качестве набора данных использовался Spotify Tracks Dataset с платформы Kaggle [2]. В данных содержится информация о 114 жанрах, по 1000 треков каждого. Мной были решены задачи классификации для **10 жанров** ('acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 'black-metal', 'bluegrass', 'blues', 'brazil') и **20 жанров** ('acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 'black-metal', 'bluegrass', 'blues', 'brazil', 'breakbeat', 'british', 'cantopop', 'chicago-house', 'children', 'chill', 'classical', 'club', 'comedy', 'country').

В качестве генератора эмбеддингов использовалась модель music2vec-v1 [1]. Модель сравнима по возможностям с текущими state-of-the-art алгоритмами, но при этом состоит из меньшего количестка параметров и позволяет легче производить перенос знаний.

![image](https://user-images.githubusercontent.com/29786176/213944658-cdc8fc5b-d86c-4a54-8425-cc88f97c1536.png)

Модель обучена по принципам самообучения модели (self-supervised learning). Teacher Model и Student Model имеют общую архитектуру и веса Teacher Model обнавляются как экспоненциальная скользящая средняя весов Student Model. Student Model получает на вход частично скрытые входные данные и оценивает по ним среднее верхних $K$ выходных слоёв трансформера в Teacher Model. При этом Teacher Model принимает на вход полные входные данные и возвращает соответствующие целевые переменные перед обучением.

Для получения более точных деталей работы music2vec-v1 можно обратиться к работе data2vec[3], так как в первой активно используются наработки последней.

## Результаты обучения

Более подробные результаты представлены на странице с экспериментами на wandb: https://wandb.ai/corgi-team/genre-classification. Для (практически) каждого запуска эксперимента можно посмотреть соответствующую архитектуру модели, открыв вкладку Logs для соответсвующего запуска.

![W B Chart 1_23_2023, 2_32_47 AM](https://user-images.githubusercontent.com/29786176/213946511-1c91bbe8-94b1-4f08-8772-a1ccc272215b.png)

![W B Chart 1_23_2023, 2_28_24 AM](https://user-images.githubusercontent.com/29786176/213946292-6604041a-11c5-4653-8db8-35ea917cbea8.png)

![W B Chart 1_23_2023, 2_36_00 AM](https://user-images.githubusercontent.com/29786176/213946569-7b9da71a-a8c0-4bf6-962e-ff93f85a98a8.png)

![W B Chart 1_23_2023, 2_35_46 AM](https://user-images.githubusercontent.com/29786176/213946564-57472e71-9c2b-4d95-89de-9eb457c62cf6.png)

Выше представлены результаты для задачи классификации **10 жанров**. Были использованы различные размеры выборок для обучения (на изображении по 2 лучших представителя для каждой выборки). scarlet-galaxy-47 и lyroc-snowball-56 (280 треков в обучающей выборке), flowing-hill-87 и noble-jazz-103 (560 треков), clean-spaceship-111 и resilient-firefly-108 (840 треков), serene-star-113 и misunderstood-salad-115 (1120 треков). Для каждого количества треков вручную подбиралась наилучшая архитектура. По данным графиков можно сделать предположение о том, что обучение на увеличеном наборе обучающих данных помогает добиться более высокой точности модели. При этом, по графиком заметно, что проблема выраженного переобучения остаётся при различныъ объёмах обучающей выборки (в проводимых экспериментах не получилось добиться падения переобучения при использовании DropOut и уменьшения архитектуры сети без потери в точности модели).

![W B Chart 1_23_2023, 2_55_12 AM](https://user-images.githubusercontent.com/29786176/213947316-169308b6-938d-4ac8-9fa9-b99f2deeb177.png)

![W B Chart 1_23_2023, 2_55_01 AM](https://user-images.githubusercontent.com/29786176/213947318-635b060c-8c46-4b88-9eeb-8966bebc15ee.png)

![W B Chart 1_23_2023, 2_55_31 AM](https://user-images.githubusercontent.com/29786176/213947319-bc0c8828-b234-42ae-bd55-a51d05de7163.png)

![W B Chart 1_23_2023, 2_55_22 AM](https://user-images.githubusercontent.com/29786176/213947315-1492457a-64fd-4775-8a54-a8ea19d4ba03.png)

Выше представлены результаты для задачи классификации **20 жанров**. Для fiery-sun-120 использовалась схожая с serene-star-113 архитектура за исключанием изменения вероятности DropOut'а на последнем слое с 0.6 на 0.7. На графиках валидации можно заметить, что модель лучше обучилась относительно величины функции потерь в сравнении со случаем из **10 жанров**. При этом, увеличение количество классов для классификации сказалось на падении F1-метрики. Использование батч-нормализации между полносвязными слоями и увеличенныйы DropOut на последнем слое до 0.8 позволил повысить качество модели до F1-macro ~0.5 на валидационной выборке.

## Источники
1. Li Y. et al. Map-music2vec: A simple and effective baseline for self-supervised music audio representation learning //arXiv preprint arXiv:2212.02508. – 2022.
2. Spotify Tracks Dataset (https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
3. Baevski A. et al. Data2vec: A general framework for self-supervised learning in speech, vision and language //arXiv preprint arXiv:2202.03555. – 2022.
