# ICPR_UT
The solutions ranked 11th and 16th in Track 5 of the NVIDIA AI City Challenge.

# Solution pipelines
1. Download the training_videos from [track5 of AI CIty Challenge](http://www.aicitychallenge.org/2024-track5-download/)
2. Extracts images from Track 5 training_videos:
- bash ./run_extract_train_frames.sh (Please see [AICITY2023_Track5_DVHRM] (https://github.com/cmtsai2023/AICITY2023_Track5_DVHRM)
3. Convert gt.txt to yolo txt:
- python GTxywh2yolo.py (Please see [AICITY2023_Track5_DVHRM] (https://github.com/cmtsai2023/AICITY2023_Track5_DVHRM)
4. Uses BF-YOLOv7 model to train the nine classes Helmet detector with 100 training videos and 100 validation videos:
- python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train_aux.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 8 --data Helmet/Helmet100.yaml --img 1920 1920 --cfg Helmet/BF-YOLOv7-Helmet.yaml --weights '' --project Helmet --name BF-YOLOv7-Helmet --hyp data/hyp.scratch.p6.yaml --epochs 350
5. Uses PRB-FPN6-MSP model to fine tune the nine classes Helmet detector based on BF-YOLOv7_best.pt with 100 training videos and 100 validation videos:
- python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train_aux.py --workers 32 --device 0,1 --sync-bn --batch-size 12 --data Helmet/Helmet100.yaml --img 1920 1920 --cfg Helmet/PRB-FPN6-MSP-Helmet.yaml --weights Helmet/BF-YOLOv7_best.pt --project Helmet --name PRB-FPN6-MSP-Helmet --hyp data/hyp.scratch.p6.yaml --epochs 100
6. Download the testing_videos from [track5 of AI CIty Challenge](http://www.aicitychallenge.org/2024-track5-download/)
7. Extracts images from Track 5 testing_videos: 
- bash ./run_extract_test_frames.sh (Please see [AICITY2023_Track5_DVHRM] (https://github.com/cmtsai2023/AICITY2023_Track5_DVHRM)
8. Rank 16: Test BF-YOLOv7_best.pt nine classes Helmet detector
- Download [BF-YOLOv7_best.pt nine classes Helmet detector model](https://drive.google.com/file/d/10UP4De9QZjthfObeeXsMolqZ0PYnnaDA/view?usp=sharing)
- python detect_Helmet.py --source /data/aicity2024/T5/track5_test/test100/images --weights Helmet/BF-YOLOv7_best.pt --conf 0.0001 --iou-thres 0.5 --img-size 1920 --device 3
9. Rank 11: Test PRB-FPN6-MSP-FT_best.pt nine classes Helmet detector
- Download [PRB-FPN6-MSP-FT_best.pt nine classes Helmet detector model](https://drive.google.com/file/d/18tv8Cvu0VvpUaajZSK--Y_mIKe6K3jHj/view?usp=sharing)
- python detect_Helmet.py --source /data/aicity2024/T5/track5_test/test100/images --weights Helmet/PRB-FPN6-MSP-FT_best.pt --conf 0.0001 --iou-thres 0.4625 --img-size 1920 --device 3
# Environment
Please refer to [YOLOv7](https://github.com/WongKinYiu/yolov7) Installation
