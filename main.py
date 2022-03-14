from os.path import isfile, join
from pickle import TRUE
from dotenv import load_dotenv
from os import listdir
from tqdm import tqdm
import pytesseract
import numpy as np
import pandas as pd
import json
import cv2
import csv
import re
import os

# PyTesseract CMD location
load_dotenv()
PYTESSERACT_DIR = os.getenv('PYTESSERACT_DIR')
pytesseract.pytesseract.tesseract_cmd = PYTESSERACT_DIR


def find_valid_date(date, month, year):
    '''
        # find_valid_date(int, int, int) -> (int, int, int)
        #    Given a date, month and year, the method returns the same if it is valid, else returns the previous valid date
        #    Ex 00: find_valid_date(2,1,2022) => (2,1,2022)
        #    Ex 01: find_valid_date(-2,1,2022) => (29,12,2021)
    '''
    if date < 1:
        if month == 2 or month == 4 or month == 6 or month == 9 or month == 11:
            date += 31
            month -= 1
            return (date, month, year)

        if month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
            date += 30
            month -= 1
            return (date, month, year)

        if month == 1:
            date += 31
            month = 12
            year -= 1
            return (date, month, year)

        if month == 3:
            month = 2
            if year % 400 == 0 or year % 100 != 0 and year % 4 == 0:
                date += 29
            else:
                date += 28
        return (date, month, year)
    return (date, month, year)


def find_tot_minutes(img):
    '''
        # find_tot_minutes(numpy.ndarray) -> float
        #    Given an image ndarray, this method extracts the text in image and returns the Y-Axis max time from the chart
    '''
    text = pytesseract.image_to_string(img)
    for i in range(6, 9):
        time = text.split('\n')[i]
        if 'hr' in time:
            hrs = re.findall(r'\d+(?:\.\d+)?', time)
            # print('\nHours: ', hrs)
            return float(hrs[0]) * 60
        elif 'min' in time:
            mins = re.findall(r'\d+(?:\.\d+)?', time)
            # print('\nMinutes: ', mins)
            return float(mins[0])


def write_into_json(data):
    '''
        # write_into_jsaon(dictionary) -> None
        #    Writes the dictionary parameter into a JSON file
    '''
    print('-> Writing data into data.json file...')
    json_dump = json.dumps(data)
    open('data.json', 'w').write(json_dump)
    print('==> Data written into data.json successfully')


def write_into_csv(data):
    '''
        # write_into_csv(dictionary):
        #    Flattens the dictionary parameter and writes it into a CSV file
    '''
    print('-> Writing data into data.csv file...')

    data_csv = pd.json_normalize(data, sep='-').to_dict(orient='records')[0]

    with open('data.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in data_csv.items():
            writer.writerow([key, value])

    print('==> Data written into data.csv successfully')


def main():
    print('-> Scanning Directory...')
    count = 0
    files = [file for file in listdir(
        './charts') if isfile(join('./charts', file))]

    print(f'==> Found {len(files)} charts')
    print('-> Extracting data from charts...')

    data = {}
    for file in tqdm(files):
        img = cv2.imread(f'./charts/{file}')
        tot_minutes = find_tot_minutes(img)

        if type(tot_minutes) is float:
            # Region of Interest (roi) is the chart part on the image
            roi = img[425:950, 0:1080]

            # Grayscale version of roi
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Edges detected using Canny from cv2
            edges = cv2.Canny(gray, 150, 350, apertureSize=3)

            # Using HoughLinesP and edges from the image the lines are detects and extracted
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 55, minLineLength=20)

            count += 1
            file = file.split('-')
            year = int(file[0][-4:])
            month = int(file[1])
            date = int(file[2])

            # Each line in lines represent the Y-Axis of bars in the chart.
            # The threshold and minLineLength parameters in Canny and HoughLinesP are optimized for the best results.
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 != x2 and y1 == y2 and y1 != 437 and y1 != 436 and y1 != 435:
                    if year not in data:
                        data[year] = {i: {} for i in range(1, 13)}

                    if x1 < 174:
                        m_date, m_month, m_year = find_valid_date(
                            date-6, month, year)
                        data[m_year][m_month][m_date] = round(
                            (437-y1) * tot_minutes / 408)
                    elif x1 >= 174 and x1 < 297:
                        m_date, m_month, m_year = find_valid_date(
                            date-5, month, year)
                        data[m_year][m_month][m_date] = round(
                            (437-y1) * tot_minutes / 408)
                    elif x1 >= 297 and x1 < 423:
                        m_date, m_month, m_year = find_valid_date(
                            date-4, month, year)
                        data[m_year][m_month][m_date] = round(
                            (437-y1) * tot_minutes / 408)
                    elif x1 >= 423 and x1 < 549:
                        m_date, m_month, m_year = find_valid_date(
                            date-3, month, year)
                        data[m_year][m_month][m_date] = round(
                            (437-y1) * tot_minutes / 408)
                    elif x1 >= 549 and x1 < 647:
                        m_date, m_month, m_year = find_valid_date(
                            date-2, month, year)
                        data[m_year][m_month][m_date] = round(
                            (437-y1) * tot_minutes / 408)
                    elif x1 >= 647 and x1 < 799:
                        m_date, m_month, m_year = find_valid_date(
                            date-1, month, year)
                        data[m_year][m_month][m_date] = round(
                            (437-y1) * tot_minutes / 408)
                    else:
                        m_date, m_month, m_year = find_valid_date(
                            date, month, year)
                        data[m_year][m_month][m_date] = round(
                            (437-y1) * tot_minutes / 408)

    print(
        f'==> Extraction Completed for {count} charts out of {len(files)} charts')
    print('==> Data Dump:')
    print(json.dumps(data, sort_keys=True, indent=4))

    write_into_json(data)
    write_into_csv(data)


if __name__ == '__main__':
    main()
