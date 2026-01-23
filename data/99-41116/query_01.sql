-- Выборка клиентов с потенциальным риском
SELECT client_id, status, score
FROM analytics.client_risk
WHERE status = 'HIGH'
ORDER BY score DESC;

-- Небезопасный пример: удаление без WHERE
DELETE FROM analytics.client_risk;
