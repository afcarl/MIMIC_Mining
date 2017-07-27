import csv
import StringIO

notes_filename = "/Users/macbook/Desktop/corpora/MIMIC/NOTEEVENTS_example.csv"

new_only_notes_filename = "/Users/macbook/Desktop/corpora/MIMIC/notes.txt"

complete_mimic_filename = "/Users/macbook/Desktop/corpora/MIMIC/NOTEEVENTS.csv"


start_string_1 = "history of present illness"
end_strings_list1 = ["past medical history:", "physical exam:", "allergies", "physical examination", "past medical history"]

start_string_2 = "past medical history"
end_strings_list2 = ["ADMITTING MEDICATIONS:", "MEDICATIONS AT HOME:", "Social History", "ALLERGIES", "Medications:", "PAST SURGICAL HISTORY:"]


def attempt_find_section(text, start_string, stop_list):

    section = ""
    low_text = text.lower()
    start_index = low_text.find(start_string)
    if start_index > -1:
        end_indices = []
        for stop_indicator in stop_list:
            end_index = low_text.find(stop_indicator.lower())
            if end_index > -1:
                end_indices.append(end_index)
        if len(end_indices) > 0:
            min_end_index = min(end_indices)
            section = low_text[start_index: min_end_index]
        else:
            section = low_text[start_index:]
    return section

with open(new_only_notes_filename, "a") as new_file:
    with open(complete_mimic_filename) as notes_file:
        data = notes_file.read()
        f=StringIO.StringIO(data)   #used just to simulate a file. Use your file here...
        reader = csv.reader(f)
        for line in reader:
            line = [x.strip() for x in line]
            if len(line):                                                                                
                new_file.write("<START_NOTE> ")
                new_file.write("ID: " + line[1] + "_" + line[2] + "_" + line[3] + " ")
                new_file.write("Section1: " + attempt_find_section(line[10], start_string_1, end_strings_list1) + " ")
                new_file.write("Section2: " + attempt_find_section(line[10], start_string_2, end_strings_list2) + " ")
                new_file.write("<END_NOTE> \n")


'''
# Use this to check and count the lines in a file
count = 0
with open(new_only_notes_filename, "a") as new_file:

    with file(notes_filename) as notes_file:

        for line in notes_file:
            line_parts = line.split(",")
            note_parts = line_parts[10:]
            complete_note = ",".join(note_parts)
            new_file.write(complete_note + "\n")
           # print "Count", len(line_parts), count, len(note_parts), len(complete_note)
            count += 1
print "COUNT:", count
'''