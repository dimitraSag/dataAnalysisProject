USE ChinookDW



INSERT INTO DimEmployee (EmployeeId, FirstName, LastName,Country, Title)
     SELECT *
         FROM ChinookStaging.dbo.DimEmployee;

INSERT INTO DimTrack (TrackId, TrackName, Genre, Composer, AlbumName, MediaTypeName, ArtistName, Milliseconds)
SELECT 
    TrackId, 
    TrackName, 
    Genre, 
    Composer, 
    AlbumTitle, 
	MediaType, 
    Artist, 
    Milliseconds 
FROM ChinookStaging.dbo.DimTrack 
;



INSERT INTO DimDate (FullDate, Year, Month, Day, WeekDay)
SELECT FullDate, Year, Month, Day, WeekDay
	FROM ChinookStaging.dbo.DimDate 
;


INSERT INTO DimCustomer (CustomerId, FirstName, LastName, Company, City, Country)
    SELECT 
         CustomerId, FirstName, LastName, Company, City, Country
     FROM ChinookStaging.dbo.DimCustomers;

INSERT INTO FactSales (InvoiceLineId, InvoiceId, InvoiceDate, CustomerId, TrackId, Quantity, UnitPrice, InvoiceTotal, EmployeeId, BillingCountry)
SELECT 
    InvoiceLineId,                                     -- Invoice Line ID
    InvoiceId,                                          -- Invoice ID
    InvoiceDate,                                        -- Invoice Date
    CustomerId,                                         -- Customer ID
    TrackId,                                           -- Track ID
    Quantity,                                          -- Quantity
    UnitPrice,                                         -- Unit Price
    Total,                              -- Invoice Total from Invoice table
	EmployeeId,
	BillingCountry
FROM ChinookStaging.dbo.FactSales ;