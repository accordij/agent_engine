-- Пример подозрительного запроса
SELECT *
FROM raw.transactions
WHERE comment LIKE '%password%';

-- Ошибка: пропущена закрывающая скобка
SELECT count(1
FROM raw.transactions;
