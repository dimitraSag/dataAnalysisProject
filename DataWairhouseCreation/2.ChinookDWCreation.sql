--CREATE DATABASE ChinookDW;

   USE ChinookDW;


   CREATE TABLE DimCustomer (
       CustomerId INT PRIMARY KEY,
       FirstName VARCHAR(40),
       LastName VARCHAR(20),
       Company VARCHAR(80),
       City VARCHAR(40),
       Country VARCHAR(40),
   );

   CREATE TABLE DimEmployee (
       EmployeeId INT PRIMARY KEY,
       FirstName VARCHAR(20),
       LastName VARCHAR(20),
	   Country VARCHAR(20),
       Title VARCHAR(30)
   );

  CREATE TABLE DimTrack (
    TrackId INT PRIMARY KEY,
    TrackName VARCHAR(200),
    Genre VARCHAR(120),
    Composer VARCHAR(220),
    AlbumName VARCHAR(160),
    MediaTypeName VARCHAR(120),
    ArtistName VARCHAR(160),
    Milliseconds INT,
);


   CREATE TABLE DimDate (
       FullDate DATE PRIMARY KEY,
       Year INT,
       Month INT,
       Day INT,
       WeekDay VARCHAR(20)
   );

   CREATE TABLE FactSales (
    InvoiceLineId INT PRIMARY KEY,
    InvoiceId INT,
    InvoiceDate DATE,
    CustomerId INT,
    TrackId INT,
    Quantity INT,
    UnitPrice NUMERIC(10, 2),
    InvoiceTotal NUMERIC(10, 2),
	EmployeeId INT,
	BillingCountry VARCHAR(20)
);