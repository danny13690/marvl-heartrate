import os
import json
import cv2
import numpy as np
from tqdm import tqdm

import torch
import face_alignment

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import Video, HTML

#save landmarks as json file to some path
def save_landmarks(landmarks, save_path):
    if not save_path:
        return
        
    if isinstance(landmarks, list):
        landmarks_dict = {}
        for i, frame_landmarks in enumerate(landmarks):
            landmarks_dict[i] = frame_landmarks.tolist()
    else:
        landmarks_dict = {0: landmarks.tolist()}
        
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(landmarks_dict, f, ensure_ascii=False, indent=4)

#load landmarks from json files stored using above method
def load_landmarks(load_path):
    
    with open(load_path) as f:
        landmarks_dict = json.load(f)
    
    num_frames = len(landmarks_dict)
    if num_frames==1:
        landmarks = np.array(landmarks_dict['0'])
    else:
        landmarks = []
        for i in range(num_frames):
            landmarks.append(np.array(landmarks_dict[str(i)]).squeeze())
    
    return landmarks

#load a video given filepath and specific output size, note that the video output
#is NOT the same dimensions as the output frames
def load_video(video_path, output_size=(270, 480)):
    
    cap = cv2.VideoCapture(video_path)
    frames_list = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(cv2.resize(frame, output_size), cv2.COLOR_BGR2RGB)
        frames_list.append(frame)
    print(f'Video {video_path} loaded: {len(frames_list)} frames with shape {np.shape(frames_list[0])}')
    fps = cap.get(cv2.CAP_PROP_FPS)

    return frames_list, fps, Video(video_path, width=output_size[0])

def draw_landmarks(image, landmarks, plot=True, save_path=None):
    
    plot_style = {'marker': 'o', 
                  'markersize': 4, 
                  'linestyle': '-', 
                  'lw': 2}
    
    landmark_types = {'face': list(range(0, 17)),
                  'eyebrow1': list(range(17, 22)),
                  'eyebrow2': list(range(22, 27)),
                  'nose': list(range(27, 31)),
                  'nostril': list(range(31, 36)),
                  'eye1': list(range(36, 42)) + [36],
                  'eye2': list(range(42, 48)) + [42],
                  'lips': list(range(48, 60)) + [48],
                  'teeth': list(range(60, 68)) + [60]
                 }
    
    type_colors = {'face': (0.682, 0.780, 0.909, 0.5),
                  'eyebrow1': (1.0, 0.498, 0.055, 0.4),
                  'eyebrow2': (1.0, 0.498, 0.055, 0.4),
                  'nose': (0.345, 0.239, 0.443, 0.4),
                  'nostril': (0.345, 0.239, 0.443, 0.4),
                  'eye1': (0.596, 0.875, 0.541, 0.3),
                  'eye2': (0.596, 0.875, 0.541, 0.3),
                  'lips': (0.596, 0.875, 0.541, 0.3),
                  'teeth': (0.596, 0.875, 0.541, 0.4)
                  }

    plt.imshow(image)
    plt.axis('off')

    if landmarks is not None:
        
        for landmark_type in landmark_types:
            type_list = landmark_types[landmark_type]
            type_array = np.stack([landmarks[idx] for idx in type_list])
            plt.plot(type_array[:, 0], type_array[:, 1], color=type_colors[landmark_type], **plot_style)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    if plot:
        plt.show()
    else:
        plt.close()

#get predictions for a single image
def fa_image_pred(image, save_json_path=None, save_plot_path=None):
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', flip_input=False)
    preds = fa.get_landmarks(image)[-1]
    save_landmarks(preds, save_path=save_json_path)
    draw_landmarks(image, preds, plot=True, save_path=save_plot_path)
    
    return preds

#get predictions for entire video, use save_frames_dir 
def fa_video_pred(in_frames, save_json_path=None, save_frames_dir=None, batch_size=32, key_sections=False):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', flip_input=False)
    num_frames = len(in_frames)
    num_batchs = int(np.ceil(num_frames / batch_size))
    preds = []
    with tqdm(total=num_frames) as pbar:
        for i in range(num_batchs):
            batch_frames = in_frames[i*batch_size:(i+1)*batch_size]
            batch = np.stack(batch_frames)
            batch = batch.transpose(0, 3, 1, 2)
            batch = torch.Tensor(batch)
            print(type(batch))
            batch_preds = fa.get_landmarks_from_batch(batch)
            print(type(batch_preds))
            preds.extend(batch_preds)
            pbar.update(batch_size)
        
    save_landmarks(preds, save_path=save_json_path)
    
    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)
        for j, pred in enumerate(preds):
            if not pred.any(): pred = None
            save_path = os.path.join(save_frames_dir, 'frame{}'.format(j))
            if key_sections:
              visualize_key_sections(in_frames[j], pred[0], False, save_path)
            else:
              draw_landmarks(in_frames[j], pred[0], False, save_path)
    
    return preds

def compose_annotated_frames(in_frames, preds, save_dir, key_sections=False, skip=5):
  for j, pred in enumerate(preds):
    if not pred.any(): pred = None
    save_path = os.path.join(save_dir, 'frame{}'.format(j))
    if j % 30 == 0:
      if key_sections:
        visualize_key_sections(in_frames[j], pred, False, save_path)
      else:
        draw_landmarks(in_frames[j], pred, False, save_path)
  plt.close()

#can be used to convert frames from above function into video
def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

#draw annotations for key_sections only
def visualize_key_sections(single_frame, single_pred, plot=True, save_path=None):

  landmark_types = {'left': [2, 27, 29, 4, 2], 'right': [27, 14, 12, 29, 27]}
  type_colors = {'left': (1.0, 0.498, 0.055, 0.4), 'right': (0.596, 0.875, 0.541, 0.3)}

  plot_style = {'marker': 'o', 
              'markersize': 4, 
              'linestyle': '-', 
              'lw': 2}

  plt.imshow(single_frame)
  plt.axis('off')

  if len(single_pred.shape) > 2:
    for i in range(single_pred.shape[0]):
      plots_2d = {}
      for landmark_type in landmark_types:
          type_list = landmark_types[landmark_type]
          type_array = np.stack([single_pred[i,idx,:] for idx in type_list])
          plots_2d[landmark_type] = plt.plot(type_array[:, 0], type_array[:, 1], 
                                             color=type_colors[landmark_type], **plot_style)[0]
  else: 
    plots_2d = {}
    for landmark_type in landmark_types:
        type_list = landmark_types[landmark_type]
        type_array = np.stack([single_pred[idx] for idx in type_list])
        plots_2d[landmark_type] = plt.plot(type_array[:, 0], type_array[:, 1], 
                                           color=type_colors[landmark_type], **plot_style)[0]
  
  if save_path:
    plt.savefig(save_path, bbox_inches='tight')
      
  if plot:
    plt.show()
  plt.cla()

#blackout rest of video except for key section
def visualize_key_sections_masked(video_frames, fps, landmarks_path):
  landmarks = load_landmarks(landmarks_path)
  roi_landmarks = [2, 27, 14, 12, 29, 4]

  mask = np.zeros(video_frames[0].shape, dtype=np.uint8)
  roi_corners = np.stack([[landmarks[0][idx][0:2] for idx in roi_landmarks]]).astype(np.int32)
  channel_count = video_frames[0].shape[2]
  ignore_mask_color = (255,)*channel_count
  cv2.fillPoly(mask, roi_corners, ignore_mask_color)
  masked_image = cv2.bitwise_and(video_frames[0], mask)
  plt.imshow(masked_image)

def get_1D_colorstream(video_frames, fps, landmarks_path):
  landmarks = load_landmarks(landmarks_path)
  assert(len(video_frames) == len(landmarks))
  roi_landmarks = [2, 27, 14, 12, 29, 4]

  signal = []
  for i in range(len(video_frames)):
    if len(landmarks[i].shape) > 2: landmarks[i] = landmarks[i][0]

    # create mask with correct height and width, no color channel
    mask = np.zeros(video_frames[i].shape[0:2], dtype=np.uint8)
    roi_corners = np.stack([[landmarks[i][idx] for idx in roi_landmarks]]).astype(np.int32)
    cv2.fillPoly(mask, roi_corners, 1)
    mean = video_frames[i][mask>0,:].mean()
    signal.append(mean)
  return np.stack(signal)

def get_3D_colorstream(video_frames, fps, landmarks_path):
  landmarks = load_landmarks(landmarks_path)
  assert(len(video_frames) == len(landmarks))
  roi_landmarks = [2, 27, 14, 12, 29, 4]

  signal = []
  for i in range(len(video_frames)):
    if len(landmarks[i].shape) > 2: landmarks[i] = landmarks[i][0]

    # create mask
    mask = np.zeros(video_frames[i].shape[0:2], dtype=np.uint8)
    roi_corners = np.stack([[landmarks[i][idx] for idx in roi_landmarks]]).astype(np.int32)
    cv2.fillPoly(mask, roi_corners, 1)
    mean = video_frames[i][mask>0,:].mean(axis=0)
    signal.append(mean)
  return np.stack(signal)
