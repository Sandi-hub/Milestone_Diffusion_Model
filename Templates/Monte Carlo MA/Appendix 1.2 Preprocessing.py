import glob
import gzip
import os
import shutil

# Customizable inputs
destination_folder_unzip = "C:/Users/srude/OneDrive - Kühne Logistics University/Masterthesis/Daten von KN/bol_unzipped"
origin_folder = "C:/Users/srude/OneDrive - Kühne Logistics University/Masterthesis/Daten von KN/bol"
destination_folder_preprocess = "C:/Users/srude/OneDrive - Kühne Logistics University/Masterthesis/Daten von KN/bol_preprocessed"


def unzip_files():
    """
    Unzip all files
    """
    # create folder in which the unzipped files will be stored
    if not os.path.exists(
            destination_folder_unzip):
        os.mkdir(destination_folder_unzip)

    # list all existing folders
    folders = [d for d in
               os.listdir(origin_folder)
               if os.path.isdir(os.path.join(origin_folder, d))]
    for folder in folders:
        new_folder = destination_folder_unzip + folder
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        directory_string = origin_folder + folder
        os.chdir(directory_string)
        for file in glob.glob("*.gz"):
            import_file = directory_string + "/" + file
            export_file = new_folder + "/" + file[:-8] + "_unzipped.json"

            # Unzip each file
            with gzip.open(import_file, 'rb') as f_in:
                with open(export_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


def preprocess():
    """
    Preprocesses the files in order to make them readable for pythons jsons module. Inset comma after each line &
    insert brackets at beginning and end
    """
    if not os.path.exists(destination_folder_preprocess):
        os.mkdir(destination_folder_preprocess)

    old_sub_folders = [d for d in
                       os.listdir(destination_folder_unzip)
                       if os.path.isdir(os.path.join(destination_folder_unzip, d))]

    for folder in old_sub_folders:
        new_sub_folder = destination_folder_preprocess + "/" + folder
        if not os.path.exists(new_sub_folder):
            os.mkdir(new_sub_folder)
        old_sub_folder = destination_folder_unzip + "/" + folder
        os.chdir(old_sub_folder)
        for file in glob.glob("*.json"):
            old_file = old_sub_folder + "/" + file
            new_file = open(new_sub_folder + "/" + file[:-14] + "_preprocesses.json", "w")
            file_handle = open(old_file)
            line_list = file_handle.readlines()
            new_file.write("[")
            for line in line_list:
                # check for first and last line: they do not need the comma
                if line == line_list[0] or line == line_list[-1]:
                    new_file.write(line.strip())
                else:
                    # check if line is empty
                    if line:
                        new_file.write(line.strip() + "," + "\n")
            new_file.write("]")
            file_handle.close()
            new_file.close()
