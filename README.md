# InstagramAnalysis
Downloads and analyses images from instagram.

Requirements
------------
The following python packages are needed to run this script:
 - tqdm
 - numpy
 - pandas
 - opencv-python
 - requests

You will also need a suitable model, the following works well in this setting:
 - Names file: https://github.com/pjreddie/darknet/blob/master/data/coco.names
 - Config file: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-spp.cfg
 - Weights file: https://pjreddie.com/media/files/yolov3-spp.weights

Usage
-----
```
script.py [-l] [--names NAMES] [--config CONFIG] [--weights WEIGHTS] [--output OUTPUT] handle

  -l                 Use file with multiple handles
  --names NAMES      'names' file used by model
  --config CONFIG    Configuration file used by model
  --weights WEIGHTS  Weights file used by model
  --output OUTPUT    File to write output to
  handle             Handle input (handle by default or file with -l flag)
```
For example:
```
$ python script.py --names coco.names --config yolov3-spp.cfg --weights yolov3-spp.weights -l handles
```

When a file containing handles is used, each line in the file should contain a single handle without the @ symbol.
