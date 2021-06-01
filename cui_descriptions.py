import csv
import json
cui_descriptions = {}
with open("cui_descriptions.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    # next(reader, None)  # skip the headers
    for row in reader:
        cui = row[0]
        description = row[1]
        lat = row[3]
        if cui in cui_descriptions:
            if lat == "SPA":
                cui_descriptions[cui]=description
        else:
            cui_descriptions[cui]=description
print(f"{len(cui_descriptions)} codes were retrieved")
with open("cui_descriptions.json", "w", encoding="utf-8") as j:
    j.write(json.dumps(cui_descriptions, indent=2, ensure_ascii=False))