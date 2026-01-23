-- Пример динамического построения условия
SELECT *
FROM mart.operations
WHERE operation_type IN ('CASH', 'TRANSFER')
AND created_at >= '2024-01-01';

-- Ошибка: неизвестная таблица
SELECT * FROM mart.missing_table;
