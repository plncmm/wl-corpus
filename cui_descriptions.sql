SELECT 
	CUI,
	STR,
    SUI,
    LAT,
    ISPREF,
    TS,
    STT,
    TTY
INTO OUTFILE '/mysql_dumps/cui_descriptions.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
FROM
	umls_full.MRCONSO
WHERE
	(TS = 'P'
    AND ISPREF = "Y"
    AND LAT = "SPA")
    OR (TS = 'P'
    AND ISPREF = "Y"
    AND LAT = "ENG"
    AND STT = "PF")
ORDER BY
	CUI ASC