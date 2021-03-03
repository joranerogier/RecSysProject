import os
import csv

def check_dir(DIR):
    """
    Check if directory already exists.
    If not, create the directory.
    """
    if not os.path.exists(DIR):
        try:
            os.mkdir(DIR)
        except OSError:
            print ("Creation of the directory %s failed" % DIR)
        else:
            print ("Successfully created the directory %s " % DIR)
    else:
        print ("Required directory '%s' exists " % DIR)


def check_csv_path(file_path, fieldnames):
    """
    Check if path already exists or not.
    If not, create a new file and initialize with the fieldnames.
    """
    if not os.path.isfile(file_path):
        print("Creating new csv file `%s`...", file_path)
        with open(file_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(fieldnames)
        csv_file.close()