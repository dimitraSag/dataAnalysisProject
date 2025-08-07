USE ChinookDW




-- 2. Δημιουργία Foreign Keys στον Fact Table
ALTER TABLE FactSales ADD CONSTRAINT FK_FactSales_Employee
FOREIGN KEY (EmployeeID) REFERENCES DimEmployee(EmployeeID);

ALTER TABLE FactSales ADD CONSTRAINT FK_FactSales_Customer
FOREIGN KEY (CustomerId) REFERENCES DimCustomer(CustomerID);

ALTER TABLE FactSales ADD CONSTRAINT FK_FactSales_Track
FOREIGN KEY (TrackId) REFERENCES DimTrack(TrackID);

ALTER TABLE FactSales ADD CONSTRAINT FK_FactSales_Date
FOREIGN KEY (InvoiceDate) REFERENCES DimDate(FullDate);