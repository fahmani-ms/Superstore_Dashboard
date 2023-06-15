use shop;
SELECT `order`.`Order ID`, `Order Priority`,od.`Product ID`,p.Category,p.`Sub-Category`,od.Quantity,od.Sales,od.Discount,od.`Shipping Cost`,s.Country,s.`Ship Mode`,`Order Date`,s.`Ship Date`
FROM `order`
JOIN shipping s on `order`.`Order ID` = s.`Order ID`
right JOIN order_detail od ON `order`.`Order ID` = od.`Order ID`
JOIN product p on p.`Product ID` = od.`Product ID`;
