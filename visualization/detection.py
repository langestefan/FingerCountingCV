
import cv2
import os, sys

import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from time import sleep

import torch
import torchvision.transforms as transforms


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax,color='red',
                               thickness=4, display_str_list=(), use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height

  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str, fill='black', font=font)
    text_bottom -= text_height - 2 * margin


def save_with_bounding_boxes(image, boxes, scores, 
                             classes, save_dir):
    """Draws bounding boxes on image.

    Args:
        image: a PIL.Image object.
        boxes: a numpy array of shape [N, 4] containing N bounding boxes of the form
                 [ymin, xmin, ymax, xmax]. If num_classes > 1, the boxes may be
                 distributed in multiple classes.
        scores: a numpy array of shape [N] or [N, num_classes]. If num_classes > 1,
                the scores correspond to the classes.
        classes: a numpy array of shape [N] or [N, num_classes]. If num_classes > 1,
                the classes correspond to the classes.
        save_dir: a string representing the directory in which image should be saved.
    """
    # 5 colours for the bounding boxes, based on the classes 1 to 5
    colours = {1: 'red', 2: 'green', 3: 'blue', 4: 'yellow', 5: 'orange'}

    # get bounding box coordinates
    xmin = round(int(boxes[0]))
    ymin = round(int(boxes[1]))
    xmax = round(int(boxes[2]))
    ymax = round(int(boxes[3]))

    # get the class name
    digit = int(classes)

    # print(type(xmin), type(ymin), type(xmax), type(ymax))
    # image = Image.fromarray(image.numpy())

    # convert image from Tensor to PIL
    trans = transforms.ToPILImage()

    image = trans(image)


    # image = image.numpy() * 255
    # Image.fromarray(image.astype(np.uint8))

    # draw bounding boxes on image
    draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, colours[digit],
                                1, display_str_list=[str(digit)], use_normalized_coordinates=False)
    
    # save image
    image_path = os.path.join(save_dir + '_.png')
    
    image.save(image_path)
