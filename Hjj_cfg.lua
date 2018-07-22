 local imgnet_cfg = {
  class_count = 21,  -- excluding background class
  target_smaller_side = 600, --480
  scales = { 112, 224, 336, 512 },
  --scales = { 64, 128, 256, 512 },
  max_pixel_size = 1000,
  normalization = { method = 'contrastive', width = 7, centering = true, scaling = true },
  augmentation = { vflip = 0, hflip = 0.25, random_scaling = 0, aspect_jitter = 0 },
  color_space = 'yuv',
  roi_pooling = { kw = 7, kh = 7 },
  --examples_base_path = './data/training/', -- saved sequence feature path
  --flow_base_path = './data/flow_training/',--'/media/zangxh/JJH/sahasuman-bmvc2016_code-fb0de209d699/ucf_train_test_data/train_test_data/',
  examples_base_path = './data/hmdb_training_new/', -- saved sequence feature path
  flow_base_path = './data/flow_training/',--'/media/zangxh/JJH/sahasuman-bmvc2016_code-fb0de209d699/ucf_train_test_data/train_test_data/',
  background_base_path = '',
  batch_size = 1, -- snippet number
  prop_batch_size = 256,
  prop_proportion = 1,
  fastrcnn_batch_size = 128,
  fastrcnn_proportion = 1/3,
  positive_threshold = 0.7, 
  negative_threshold = 0.3,
  best_match = true,
  nearby_aversion = false,
  flow = 1, -- choose feature rgb=0,flow=1,fusion=2
  finetune = 3,
  top_n = 30, -- maxium number of paths can be keeped in buff
  enable_reg = 0,
  nms_threshold = 0.7,
  iou_threshold=0.7
}

return imgnet_cfg