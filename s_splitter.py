import os
from glob import glob
import re
import pandas as pd


kanun_case_insensitive_pattern = r"([^\n]+)\s*\n[Mm][Aa][Dd][Dd][Ee]\s+([^\s-]+)\s*-\s*(.+?)(?=\n[A-Z])"




def make_all():
    files = glob("output_txt/Kanun/**/*", recursive=True)
    files = [file for file in files if os.path.isfile(file)]

    output_folder = "output_csv"
    os.makedirs(output_folder, exist_ok=True)

    print(f"Found {len(files)} files")

    empty_files = 0

    for file in files:
        # Read the file
        all_matches = []

        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()

        matches = re.findall(kanun_case_insensitive_pattern, text)
       
        if len(matches) != 0:
            for i, match in enumerate(matches):
                all_matches.append([1, file.split('/')[-1], *match])
       
        if len(matches) == 0:
            next_re = r'(\w+)\n\s*MADDE (\d+)- (.*?)(?=\n\s*\w+|$)'
            matches = re.findall(next_re, text)
            if len(matches) != 0:    
                for i, match in enumerate(matches):
                    all_matches.append([2, file.split('/')[-1], *match])

        if len(matches) == 0:
            next_re = r'([^\n]+)\n*?MADDE\s+(\d+)\s+–\s+(\(.*?\s+[^M]+)'
            matches = re.findall(next_re, text)
            if len(matches) != 0:    
                for i, match in enumerate(matches):
                    all_matches.append([3, file.split('/')[-1], *match])

        if len(matches) == 0:
            next_re = r"Madde\s*(\d+)-\s*([^\n]+)\s*\n\s*(\d+\.\d+.+?)(?=\n\s*Madde|\Z)"
            matches = re.findall(next_re, text)
            if len(matches) != 0:    
                for i, match in enumerate(matches):
                    all_matches.append([4, file.split('/')[-1], *match])

        if len(matches) == 0:
            next_re = r"Madde\s*(\d+)\s*-\s*(.*?)\s*(?=(?:Madde|\Z))"
            matches = re.findall(next_re, text, re.DOTALL)

            output = []
            for match in matches:
                madde_num = match[0].strip()
                title = match[1].split("\n", 1)[0].strip()
                body = match[1].split("\n", 1)[1].strip() if "\n" in match[1] else ''
                output.append((title, madde_num, body))

            if len(matches) != 0:    
                for i, match in enumerate(output):
                    all_matches.append([5, file.split('/')[-1], *match])

        
        
        if len(matches) == 0:
            print(f"Empty file: {file.split('/')[-1]}")
            empty_files += 1
            continue

                        

        df = pd.DataFrame(all_matches, columns=["Type", "FileName", "Header", "Madde", "Text"])
    
        # save csv file while preserving the folder structure
        relative_path = os.path.relpath(file, "output_txt")  # Get relative path from base input dir
        csv_file_path = os.path.join(output_folder, relative_path + ".csv")  # Append .txt to the relative path
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        df.to_csv(csv_file_path, index=False)

    print(f"Empty files: {empty_files}")


make_all()