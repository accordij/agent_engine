-- Нормальная агрегация
SELECT region, COUNT(*) AS cnt
FROM mart.client_dim
GROUP BY region;

-- Потенциально токсичный пример: хардкод пароля в комментарии
-- password = 'qwerty123'
SELECT * FROM mart.client_dim;
