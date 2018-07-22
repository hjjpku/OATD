--require 'faster-rcnn.torch/utilities.lua'

function load_model(cfg, model_path, network_filename, cuda)
  cuda = true -- need gpu support

  -- get configuration & model
  local model_factory = dofile(model_path)
  local model = model_factory(cfg) -- return pnet and cnet
  
  if cuda then
    model.cnet = model.cnet:cuda()
    model.pnet.red_net = model.pnet.red_net:cuda()
    model.pnet.enc = model.pnet.enc:cuda()
    model.pnet.rpn = model.pnet.rpn:cuda()
  end
  
  -- combine parameters from pnet and cnet into flat tensors
  -- in utilities.lua
  local weights, gradient = combine_and_flatten_parameters(model.pnet.red_net, model.pnet.enc,model.pnet.rpn, model.cnet)
  
  -- init or resume training status
  local training_stats
  if network_filename and #network_filename > 0 then
    local stored = load_obj(network_filename) -- in utilities.lua
    training_stats = stored.stats
    weights:copy(stored.weights)
  end
	
  return model, weights, gradient, training_stats
end

 local imgnet_cfg = {
  class_count = 200,  -- excluding background class
  target_smaller_side = 480,
  scales = { 48, 96, 192, 384 },
  max_pixel_size = 1000,
  normalization = { method = 'contrastive', width = 7, centering = true, scaling = true },
  augmentation = { vflip = 0, hflip = 0.25, random_scaling = 0, aspect_jitter = 0 },
  color_space = 'yuv',
  roi_pooling = { kw = 6, kh = 6 },
  examples_base_path = '',
  background_base_path = '',
  batch_size = 300,
  positive_threshold = 0.6, 
  negative_threshold = 0.25,
  best_match = true,
  nearby_aversion = true
}

--model, weights, gradient, training_stats = load_model( imgnet_cfg,'Hjj_model.lua', nil, true)