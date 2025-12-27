import argparse
import sys
from pathlib import Path
import csv


#!/usr/bin/env python3
# filepath: /home/aspi/Project/nasap-fit-cpp/test.py
# Pd: concentration.csv の濃度を config.yaml の "100%concentration" を基準に百分率に変換して出力するスクリプト
fullconc=[0.0020300000000000001, 0.0013533333333333333, 0.00022555555555555556, 0.0040600000000000002]

with open('data/M9L6/concentration.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    data = list(reader)

for row in data[1:]:
    for i in range(1, len(row)):
        row[i] = str(float(row[i]) / fullconc[i-1] /10)

with open('data/M9L6/concentration_percentage.csv', mode='w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

