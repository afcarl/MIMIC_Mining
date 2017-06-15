import csv
import StringIO

notes_filename = "/Users/macbook/Desktop/corpora/MIMIC/NOTEEVENTS_example.csv"

new_only_notes_filename = "/Users/macbook/Desktop/corpora/MIMIC/notes.txt"

with open(new_only_notes_filename, "a") as new_file:
    with open(notes_filename) as notes_file:
        data = notes_file.read()
        f=StringIO.StringIO(data)   #used just to simulate a file. Use your file here...
        reader = csv.reader(f)
        for line in reader:
            line=[x.strip() for x in line]   # remove 'if x' if you want blank fields
            if len(line):
                new_file.write("<START_NOTE>" + line[10] + "<END_NOTE> \n")