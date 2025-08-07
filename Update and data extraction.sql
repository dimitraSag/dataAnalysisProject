use chinook

SELECT 
    i.InvoiceId,
    i.CustomerId,
    i.Total
FROM Invoice i
WHERE i.Total IS NULL OR i.Total = 0;


SELECT 
    il.InvoiceId,
    il.TrackId,
    il.UnitPrice,
    il.Quantity
FROM InvoiceLine il
WHERE il.InvoiceId IN (SELECT InvoiceId FROM Invoice WHERE Total IS NULL OR Total = 0);


SELECT 
    i.InvoiceId,
    SUM(il.UnitPrice * il.Quantity) AS CalculatedTotal
FROM Invoice i
JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
GROUP BY i.InvoiceId;


UPDATE Invoice
SET Total = (
    SELECT SUM(il.UnitPrice * il.Quantity)
    FROM InvoiceLine il
    WHERE il.InvoiceId = Invoice.InvoiceId
)
WHERE Total IS NULL OR Total = 0;


SELECT 
    e.EmployeeId,
    e.FirstName,
    e.LastName,
    e.HireDate,
    COUNT(DISTINCT i.InvoiceId) AS NumberOfInvoices,
    COALESCE(SUM(i.Total), 0) AS TotalRevenue,
    COUNT(DISTINCT c.CustomerId) AS NumberOfCustomers,
    CASE 
        WHEN COUNT(DISTINCT c.CustomerId) > 0 THEN 
            CAST(COALESCE(SUM(i.Total), 0) / COUNT(DISTINCT c.CustomerId) AS DECIMAL(10,2))
        ELSE 0
    END AS AverageRevenuePerCustomer
FROM Employee e
LEFT JOIN Customer c ON e.EmployeeId = c.SupportRepId
LEFT JOIN Invoice i ON c.CustomerId = i.CustomerId
LEFT JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
GROUP BY e.EmployeeId, e.FirstName, e.LastName, e.HireDate
ORDER BY TotalRevenue DESC;
