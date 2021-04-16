import csv
import cv2
import os
from enum import IntEnum
from datetime import datetime

class Dataset():
	def __init__(self, csv_path, backup_path=None, auto_backup=False):
		self.csv_path = csv_path
		self.backup_path = backup_path
		self.auto_backup = auto_backup
		self.fields, self.rows = self.open_csv()
		self.num_cols = len(self.rows[0])
		self.num_datapoints = len(self.rows)

	def open_csv(self):
		fields = []
		rows = []
		with open(self.csv_path, 'r+', encoding='utf-8-sig') as csvfile:
			csvreader = csv.reader(csvfile)
			for row in csvreader: 
				rows.append(row)
		fields = IntEnum('Fields', rows[0][1:])
		return fields, rows

	def get_fields(self):
		return self.rows[0]

	def get_field_index(self, field):
		if field not in self.rows[0]: return None
		for i in range(self.num_cols):
			if self.rows[0][i] == field: return i

	def get_val(self, index, field, type=str):
		if self.get_field_index == None: return None
		return type(self.rows[index][self.get_field_index(field)])

	def change_val(self, index, field, val):
		if self.get_field_index == None: return None
		if type(val) != str: val = str(val)
		self.rows[index][self.get_field_index(field)] = val
		self.write_csv()

	def write_csv(self):
		with open(self.csv_path, 'w+', encoding='utf-8-sig') as csvfile:
		    csvwriter = csv.writer(csvfile) 
		    csvwriter.writerows(self.rows)

	def save_backup(self, caller=""):
		if self.backup_path == None: return
		path = self.backup_path + caller + datetime.now().strftime("%Y.%m.%d.%H:%M:%S") + ".csv"
		with open(path, 'w+', encoding='utf-8-sig') as csvfile:
		    csvwriter = csv.writer(csvfile) 
		    csvwriter.writerows(self.rows)

	def add_col(self, col_name):
		if self.auto_backup: self.save_backup("add_col:" + col_name)
		if col_name in self.rows[0]: return
		self.rows[0].append(col_name)
		self.num_cols += 1
		self.fields = IntEnum('Fields', self.rows[0][1:])
		for i in range(len(self.rows)):
			if i == 0: continue
			self.rows[i].append("")
		self.write_csv()

	def delete_col(self, col_name):
		if self.auto_backup: self.save_backup("del_col:" + col_name)
		if col_name not in self.rows[0]: return
		for i in range(self.num_cols):
			if col_name == self.rows[0][i]: to_delete = i

		for i in range(len(self.rows)):
			self.rows[i].pop(to_delete)

		self.fields = IntEnum('Fields', self.rows[0][1:])
		self.num_cols -= 1
		self.write_csv()

	def rename_col(self, old_col, new_col_name):
		if self.auto_backup: self.save_backup("rename_col:" + old_col)
		if old_col not in self.rows[0]: return
		for i in range(self.num_cols):
			if rows[0][j] == old_col:
				rows[0][j] = new_col_name
		self.fields = IntEnum('Fields', self.rows[0][1:])
		self.write_csv()

	def get_indices_pos(self, column_idx, value):
		if type(value) != str: value = str(value)
		return [i+1 for i in range(self.num_datapoints) if rows[i+1][column_idx]==value]

	def get_indices_neg(self, column_idx, value):
		if type(value) != str: value = str(value)
		return [i+1 for i in range(self.num_datapoints) if rows[i+1][column_idx]!=value]

	def extract_dataset(self, indices, data_map, datatype=float):
		dataset = {}
		for k,v in data_map:
			dataset[k] = [datatype(rows[i][v]) for i in indices]
		return dataset

