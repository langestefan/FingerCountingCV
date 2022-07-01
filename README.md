# Finger counting with a webcam through real-time object detection with PyTorch

This project uses PyTorch to build an object detector that can run in realtime.

The finger detector uses transfer learning and is built around a finetuned [FCOS](https://arxiv.org/abs/1904.01355) model combined with a ResNet50 backbone. Using an object detector makes it possible to count fingers on an arbitrary number of hands (as long as they don't overlap) and we also get position information, in case this might be useful for you.

The real-time object detector code is based on this blog: [Real-time object detection with deep learning and OpenCV](https://pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/)

Since I couldn't find any object detection dataset for counting fingers I made one by scraping stock images on the internet and annotating them. If you would like access to this dataset please e-mail me at `s.e.d.lange#at#student.tue.nl` and I will send it to you.
