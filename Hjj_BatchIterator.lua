require 'image'
require 'Hjj_utilities'
require 'Anchors'
require 'Hjj_global'
require 'math'
local matio = require 'matio'
local BatchIterator = torch.class('BatchIterator')

local function randomize_order(...)
  local sets = { ... }
  for i,x in ipairs(sets) do
    if x.list and #x.list > 0 then   -- e.g. background examples are optional and randperm does not like 0 count
      x.order:randperm(#x.list)   -- shuffle
    end
    x.i = 1   -- reset index positions
  end
end

local function next_entry(set,count)
  if set.i > #set.list then
    randomize_order(set)
  end
  
  local fn = set.list[set.order[set.i]]
  set.i = set.i + 1
  return fn
  --return set.list[count*1000]
end


function BatchIterator:__init(model, training_data)
  local cfg = model.cfg
  
  -- bounding box data (defined in pixels on original image)
  self.ground_truth = training_data.ground_truth 
  self.cfg = cfg
  
  if cfg.normalization.method == 'contrastive' then
    self.normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(cfg.normalization.width))
  else
    self.normalization = nn.Identity()
  end
  
  self.anchors = Anchors.new(model.pnet.rpn, cfg.scales)
  
  -- index tensors define evaluation order
  self.training = { order = torch.IntTensor(), list = training_data.training_set }
  self.validation = { order = torch.IntTensor(), list = training_data.validation_set }
  self.background = { order = torch.IntTensor(), list = training_data.background_files or {} }
  
  randomize_order(self.training, self.validation, self.background)
end
  
function BatchIterator:processRoIs(rois, img_size) -- transform the rois rect according to the transformation of faster-rcnn input
  local cfg = self.cfg
  local max_pixel_size = cfg.max_pixel_size
  local short_side, long_side = 0, 0
  if img_size[1] > img_size[2] then
    short_side = img_size[2]
    long_side = img_size[1]
  else
    short_side = img_size[1]
    long_side = img_size[2]
  end
  local resize_scale = cfg.target_smaller_side/short_side
  if resize_scale * long_side > max_pixel_size then
    resize_scale = max_pixel_size/long_side
  end
  img_size[1] = img_size[1] * resize_scale
  img_size[2] = img_size[2] * resize_scale

  if rois then
    for i=1,#rois do
      local roi = rois[i]
      roi.rect = roi.rect:scale(resize_scale) 
    end
  end

  return img_size, rois
end

function BatchIterator:nextTraining(count)
  local cfg = self.cfg
  local batch = {}
  count = count or cfg.batch_size
  
  -- use local function to allow early exits in case of to image load failures
  local function try_add_next()
    local seq = {}
    
    seq.img_set = {}

    local fn = next_entry(self.training,count) -- return seq feature file name

    print(fn)
    if cfg.flow == 1 then

      local flow_fn = string.gsub(fn,'.t7', '_flow.t7')
      seq.seq_fea = func_get_seq_fea(flow_fn,cfg.flow_base_path) -- in Hjj_utilities.lua
      --[[
      local info = self.ground_truth[fn][1].info
      local frm_id = info.frm
      local video_path = cfg.flow_base_path .. info.vn .. '/flow_img/'
      local seq_fea = {}
      for i=1, ConvGRU_rho do
        local fea_path = video_path .. frm_id +i-1 .. '_flow.mat'
        local conv_fea = matio.load(fea_path)
        conv_fea = conv_fea.conv_fea
        conv_fea.type(torch.Tensor)
        conv_fea = conv_fea:cuda()
        table.insert(seq_fea, conv_fea)
      end
      seq.seq_fea = func_table_2_tensor(seq_fea)
      --]]
    elseif cfg.flow == 2 then
      local rgb_fea = func_get_seq_fea(fn,cfg.examples_base_path) 
      local flow_fn = string.gsub(fn,'.t7', '_flow.t7')
      local flow_fea = func_get_seq_fea(flow_fn,cfg.flow_base_path)
      seq.seq_fea = func_fea_overlay(rgb_fea, flow_fea)
    else
      seq.seq_fea = func_get_seq_fea(fn,cfg.examples_base_path) -- in Hjj_utilities.lua
    end


    for i=1, ConvGRU_rho do -- ConvGRU_rho is global variant in Hjj_global
      local img = {}
      local img_size  = self.ground_truth[fn][i].img_size:clone()

      local rois = deep_copy(self.ground_truth[fn][i].rois)   -- copy RoIs ground-truth data (will be manipulated) in Hjj_utilities
      
      img_size, rois = self:processRoIs(rois, img_size) -- transform the rois rect according to the transformation of faster-rcnn input
      --print(img_size)

      img.size = img_size
      img.rois = rois

      if cfg.finetune == 0 then -- get anchors to train rpn 
        -- find positive examples
        local img_rect = Rect.new(0, 0, img_size[2], img_size[1])

        local t1 = os.clock()
        img.positive = self.anchors:findPositive(rois, img_rect, cfg.positive_threshold, cfg.negative_threshold, cfg.best_match)
        local t2 = os.clock()
        --print('get pos samples = ' .. t2-t1)
        -- random negative examples
        img.negative = self.anchors:sampleNegative(img_rect, rois, cfg.negative_threshold, math.max(#img.positive/cfg.prop_proportion,16) )


        if cfg.nearby_aversion then
          local nearby_negative = {}
          -- add all nearby negative anchors
          for i,p in ipairs(img.positive) do
            local cx, cy = p[1]:center()
            local nearbyAnchors = self.anchors:findNearby(cx, cy)
            for i,a in ipairs(nearbyAnchors) do
              if Rect.IoU(p[1], a) < cfg.negative_threshold then
                table.insert(nearby_negative, { a })
              end
            end
          end
          
          local c = math.min(#img.positive, count)
          shuffle_n(nearby_negative, c)
          for i=1,c do
            table.insert(img.negative, nearby_negative[i])
            --count = count + 1
          end
        end

        -- debug boxes

        if false then
          local matio = require 'matio'

          local neg_position = torch.Tensor(#img.negative,4):fill(0)
          local pos_position = torch.Tensor(#img.positive,4):fill(0)
          local roi_position = torch.Tensor(#rois,4):fill(0)
          
          for i=1,#img.negative do
            neg_position[i][1] = img.negative[i][1].minY
            neg_position[i][2] = img.negative[i][1].minX
            neg_position[i][3] = img.negative[i][1].maxY
            neg_position[i][4] = img.negative[i][1].maxX
          end
          
          for i=1,#img.positive do
            pos_position[i][1] = img.positive[i][1].minY
            pos_position[i][2] = img.positive[i][1].minX
            pos_position[i][3] = img.positive[i][1].maxY
            pos_position[i][4] = img.positive[i][1].maxX
          end
          print(rois)
          for i=1,#rois do
            roi_position[i][1] = rois[i].rect.minY
            roi_position[i][2] = rois[i].rect.minX
            roi_position[i][3] = rois[i].rect.maxY
            roi_position[i][4] = rois[i].rect.maxX
          end

          print(self.ground_truth[fn][i].info)

          matio.save('debug_box.mat',{neg=neg_position,pos=pos_position,roi=roi_position})
          print(string.format("'%s' (%dx%d); p: %d; n: %d", fn, img_size[2], img_size[1], #img.positive, #img.negative))

          os.exit()
        end
      end

      --print(string.format("'%s' (%dx%d); p: %d; n: %d", fn, img_size[2], img_size[1], #img.positive, #img.negative))
      table.insert(seq.img_set, img)
    end -- for i=1, ConvGRU_rho do
  
    table.insert(batch, seq)
    
    --return count
  end --local fucntion try_add_next()
  
  while count > 0 do
    try_add_next()
    count = count - 1
  end
  
  return batch
end

function BatchIterator:nextValidation(count)
  local cfg = self.cfg
  local batch = {}
  count = count or 1
  
  -- use local function to allow early exits in case of to image load failures
  while count > 0 do
    local fn = next_entry(self.validation)
  
    -- load image, wrap with pcall since image net contains invalid non-jpeg files
    local status, img = pcall(function () return load_image(fn, cfg.color_space, cfg.examples_base_path) end)
    if not status then
      -- pcall failed, corrupted image file?
      print(string.format("Invalid image '%s': %s", fn, img))
       goto continue
    end

    local img_size = img:size()
    if img:nDimension() ~= 3 or img_size[1] ~= 3 then
      print(string.format("Warning: Skipping image '%s'. Unexpected channel count: %d (dim: %d)", fn, img_size[1], img:nDimension()))
      goto continue
    end 
    
    local rois = deep_copy(self.ground_truth[fn].rois)   -- copy RoIs ground-truth data (will be manipulated, e.g. scaled)
    local img, rois = self:processImage(img, rois)
    img_size = img:size()        -- get final size
    if img_size[2] < 128 or img_size[3] < 128 then
      print(string.format("Warning: Skipping image '%s'. Invalid size after process: (%dx%d)", fn, img_size[3], img_size[2]))  
      goto continue
    end
      
    table.insert(batch, { img = img, rois = rois })
  
    count = count - 1
    ::continue::
  end
  
  return batch  
end
