import csv
import cv2
import os
from enum import IntEnum

def open_csv(csv_path):
	fields = []
	rows = []
	with open(csv_path, 'r+', encoding='utf-8-sig') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader: 
			rows.append(row)
	fields = IntEnum('Fields', rows[0][1:])
	return fields, rows

def initial_hr_loop(video_dir, fields, rows):
	for filename in os.listdir(video_dir):

		#open using csv
	    cap = cv2.VideoCapture(video_dir + filename)

	    #get index of file (random video load order)
	    index = int(filename[:filename.find('.')])

	    #calculate appropriate fields
	    fps = cap.get(cv2.CAP_PROP_FPS)
	    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	    length = frame_count / fps
	    beat_count = int(rows[index][fields.beat_count])

	    #insert into rows
	    rows[index][fields.vid_length_seconds] = str(length)
	    rows[index][fields.vid_length_frames] = str(frame_count)
	    rows[index][fields.vid_fps] = str(fps)
	    rows[index][fields.adjusted_hr_hertz] = str(beat_count / length)
	    rows[index][fields.adjusted_hr_bpm] = str(beat_count * 60 / length)

def write_csv(csv_path, rows):
	with open(csv_path, 'w+', encoding='utf-8-sig') as csvfile:
	    csvwriter = csv.writer(csvfile) 
	    csvwriter.writerows(rows)

