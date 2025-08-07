--CREATE DATABASE ChinookStaging
--GO

USE ChinookStaging
GO

DROP TABLE IF EXISTS ChinookStaging.dbo.FactSales;
DROP TABLE IF EXISTS ChinookStaging.dbo.DimCustomers;
DROP TABLE IF EXISTS ChinookStaging.dbo.DimTrack;
DROP TABLE IF EXISTS ChinookStaging.dbo.DimEmployee;
DROP TABLE IF EXISTS ChinookStaging.dbo.DimDate;


--1. get data FROM Employee


SELECT EmployeeID, FirstName, LastName, Country, Title
INTO ChinookStaging.dbo.DimEmployee
FROM chinook.dbo.Employee


--2 get FROM Customers


SELECT  CustomerID, FirstName, LastName, Company, Email, Phone, Address, City, PostalCode, Country
INTO ChinookStaging.dbo.DimCustomers
FROM chinook.dbo.Customer


--3  get FROM Track

SELECT  TrackID, Track.Name as TrackName, Composer, Title as AlbumTitle, Genre.Name as Genre , 
Milliseconds, Bytes, MediaType.Name as MediaType, Artist.Name as Artist
INTO ChinookStaging.dbo.DimTrack
FROM chinook.dbo.Track
INNER JOIN chinook.[dbo].Album
    ON chinook.[dbo].Track.AlbumId = chinook.[dbo].Album.AlbumId
INNER JOIN chinook.[dbo].Artist
    ON chinook.dbo.Album.ArtistId = chinook.[dbo].Artist.ArtistId
INNER JOIN chinook.dbo.Genre
	ON chinook.dbo.Track.GenreId = chinook.[dbo].Genre.GenreId
INNER JOIN chinook.dbo.MediaType
	ON chinook.dbo.Track.MediaTypeId = chinook.dbo.MediaType.MediaTypeId



--4  get FROM Invoice

SELECT InvoiceLineId, chinook.dbo.Invoice.InvoiceId, InvoiceDate, Total, chinook.[dbo].Invoice.CustomerId, TrackId, EmployeeID ,
UnitPrice, Quantity, BillingCountry
INTO ChinookStaging.dbo.FactSales
FROM chinook.[dbo].Invoice
INNER JOIN chinook.[dbo].InvoiceLine
    ON chinook.[dbo].Invoice.InvoiceId = chinook.[dbo].InvoiceLine.InvoiceId
INNER JOIN chinook.dbo.Customer
	ON chinook.dbo.Invoice.CustomerId = chinook.dbo.Customer.CustomerId
INNER JOIN chinook.dbo.Employee
	ON chinook.dbo.Customer.SupportRepId = chinook.dbo.Employee.EmployeeId


--Create DimDate
	
SELECT
    CONVERT(INT, FORMAT(i.InvoiceDate, 'yyyyMMdd')) AS DateId,  -- Format as YYYYMMDD
    i.InvoiceDate AS FullDate,
    YEAR(i.InvoiceDate) AS Year,
    MONTH(i.InvoiceDate) AS Month,
    DAY(i.InvoiceDate) AS Day,
    DATENAME(WEEKDAY, i.InvoiceDate) AS WeekDay
INTO DimDate
FROM chinook.dbo.Invoice i
GROUP BY i.InvoiceDate;















--5 date dimension


--SELECT MIN(InvoiceDate) minDate, MAX(InvoiceDate) maxDate FROM FactSales

-- SELECT
--   MIN(orderdate), MIN(shippeddate), MIN(requireddate),
--   MAX(orderdate), MAX(shippeddate), MAX(requireddate)
-- FROM orders



