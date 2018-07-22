require 'torch'
require 'pl'
require 'optim'
require 'image'
require 'nngraph'
require 'cunn'
require 'nms'
require 'gnuplot'
require 'cutorch'

require 'Hjj_utilities'
require 'Anchors'
require 'Hjj_BatchIterator'
require 'Hjj_objective'
require 'Hjj_global'
require 'Hjj_New_Detector'
require 'Rect'
local matio = require 'matio'
--require 'Detector'


-- command line options
cmd = torch.CmdLine()
cmd:addTime()

cmd:text()
cmd:text('Training a convnet for region proposals')
cmd:text()

cmd:text('=== Training ===')
cmd:option('-cfg', 'Hjj_cfg.lua', 'configuration file')
cmd:option('-model', 'Hjj_model.lua', 'model factory file')
-- ######################## ucf101 ############################
--[[
cmd:option('-name', 'ucf101', 'experiment name, snapshot prefix') 
cmd:option('-train', 'data/training/ground_truth.t7', 'training data file name')
cmd:option('-test_list', '/media/zangxh/JJH/sahasuman-bmvc2016_code-fb0de209d699/ucf_train_test_data/test_video_list.txt', 'test data file name')
cmd:option('-test_data', '/media/zangxh/JJH/sahasuman-bmvc2016_code-fb0de209d699/ucf_train_test_data/train_test_data/', 'test data path')--]]

-- ####################### HMDB ################################
----[[
cmd:option('-name', 'hmdb', 'experiment name, snapshot prefix') 
cmd:option('-train', 'data/hmdb_training_new/ground_truth.t7', 'training data file name')
cmd:option('-test_list', '/media/zangxh/JJH/sahasuman-bmvc2016_code-fb0de209d699/hmdb21_train_test_data/new_list.txt', 'test data file name')
cmd:option('-test_data', '/media/zangxh/JJH/sahasuman-bmvc2016_code-fb0de209d699/hmdb21_train_test_data/train_test_data/', 'test data path')
--]]

-- ####################### ucf-sports ################################
cmd:option('-ucf_sp_test_list', '/media/zangxh/JJH/sahasuman-bmvc2016_code-fb0de209d699/ucf-sports/ucf-sports/test_name_list.txt', 'test data file name')
cmd:option('-ucf_sp_test_data', '/media/zangxh/JJH/sahasuman-bmvc2016_code-fb0de209d699/ucf-sports/ucf-sports/UCF_SPORTS_IMGS_FLOWS/', 'test data pa')
cmd:option('-restore', '', 'network snapshot file name to load')
cmd:option('-restore2', '', 'network snapshot file name to load')
cmd:option('-validate_name','4_ucf101_004500/')
cmd:option('-snapshot', 15000, 'snapshot interval')
cmd:option('-plot', 500, 'plot training progress interval')
cmd:option('-lr', 5E-5, 'learn rate')
cmd:option('-rms_decay', 0.9, 'RMSprop moving average dissolving factor')
cmd:option('-opti', 'rmsprop', 'Optimizer')
cmd:option('-save_name', '12', 'model version')
cmd:option('-enable_reg',1,'cfg')
cmd:option('-iou_thd',0.7,'iou_thd')
cmd:option('-nms_thd',0.7,'nms_thd')
cmd:option('-flow',0,'flow')
cmd:option('-batch_num',1,'test batch num')
cmd:option('-batch_id',1, 'test batch id')


cmd:text('=== Misc ===')
cmd:option('-threads', 8, 'number of threads')
cmd:option('-gpuid', 0, 'device ID (CUDA), (use -1 for CPU)')
cmd:option('-seed', 0, 'random seed (0 = no fixed seed)')

print('Command line args:')
local opt = cmd:parse(arg or {})
print(opt)

print('Options:')
local cfg = dofile(opt.cfg)
cfg.enable_reg = opt.enable_reg or cfg.enable_reg
cfg.nms_threshold = opt.nms_thd or cfg.nms_threshold
cfg.iou_threshold = opt.iou_thd or cfg.iou_threshold
cfg.flow = opt.flow or cfg.flow
cfg.batch_num = opt.batch_num
cfg.batch_id = opt.batch_id
print(cfg)

cutorch.setDevice(DEFAULT_GPU) 


-- system configuration
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)
if opt.seed ~= 0 then
  torch.manualSeed(opt.seed)
  cutorch.manualSeed(opt.seed)
end

function plot_training_progress(prefix, stats)
  local fn = prefix .. '_progress_cls.png'
  gnuplot.pngfigure(fn)
  gnuplot.title('Traning progress over time')
  
  local xs = torch.range(1, #stats.pcls)
  
  gnuplot.plot(
    { 'pcls', xs, torch.Tensor(stats.pcls), '-' },
    { 'dcls', xs, torch.Tensor(stats.dcls), '-' }
  )
 
  gnuplot.axis({ 0, #stats.pcls, 0, 5 })
  gnuplot.xlabel('iteration')
  gnuplot.ylabel('loss')
  
  gnuplot.plotflush()

  fn = prefix .. '_progress_reg.png'
  gnuplot.pngfigure(fn)
  gnuplot.title('Traning progress over time')
  
  local xs = torch.range(1, #stats.pcls)
  
  gnuplot.plot(
    { 'preg', xs, torch.Tensor(stats.preg), '-' },
    { 'dreg', xs, torch.Tensor(stats.dreg), '-' }
  )
 
  gnuplot.axis({ 0, #stats.pcls, 0, 5 })
  gnuplot.xlabel('iteration')
  gnuplot.ylabel('loss')
  
  gnuplot.plotflush()


end

function load_model(cfg, model_path, network_filename, t_d_flag)
  -- t_d_flag -1->detect rpn; 0->detect fastrcnn; 1->train

  -- get configuration & model
  local model_factory = dofile(model_path)
  local model = model_factory(cfg) -- return pnet and cnet
  
  ----[[
  -- set default GPU
  if t_d_flag == 1 and cfg.finetune == 2 then
    cutorch.withDevice(GPU_ID, function() model.cnet = model.cnet:cuda() end)
  else
    model.cnet = model.cnet:cuda()
  end
  --model.cnet = model.cnet:cuda()
  -- set default GPU
  model.pnet.red_net = model.pnet.red_net:cuda()
  model.pnet.enc = model.pnet.enc:cuda()
  model.pnet.rpn = model.pnet.rpn:cuda()

  --]]
  
  -- combine parameters from pnet and cnet into flat tensors
  -- in utilities.lua
  local weights, gradient = combine_and_flatten_parameters(model.pnet.red_net, model.pnet.enc,model.pnet.rpn, model.cnet)
  print(gradient:size())
  --[[
  print(gradient:size())

  local weights2, gradient2 = combine_and_flatten_parameters(model.pnet.red_net, model.pnet.enc,model.pnet.rpn)

  print(gradient2:size())
  os.exit()
  --]]
  -- resume training status if given training history
  local training_stats
  if network_filename and #network_filename > 0 then
    print('load model')

    local stored = load_obj(network_filename) -- in utilities.lua
    --training_stats = stored.stats
    print(stored.weights:size())
    -- 128 dim ->3271117
    if cfg.finetune == 1 and t_d_flag == 1 then
      weights[{{1,12931917}}]:copy(stored.weights[{{1,12931917}}])
    --elseif cfg.finetune == 3 and t_d_flag == 1 then
      --weights[{{1,12931917}}]:copy(stored.weights[{{1,12931917}}])
    elseif t_d_flag == -1 then
      --weights[{{1,12931917}}]:copy(stored.weights[{{1,12931917}}])
      weights[{{1,3271117}}]:copy(stored.weights[{{1,3271117}}])
    else
       weights:copy(stored.weights)
       --weights[{{1,3271117}}]:copy(stored.weights[{{1,3271117}}])
    end
  end
--[[
  if cfg.finetune == 3 and t_d_flag == 1 then
    cutorch.withDevice(GPU_ID, function() model_2.pnet.red_net = model.pnet.red_net:clone() end)
    cutorch.withDevice(GPU_ID, function() model_2.pnet.enc = model.pnet.enc:clone() end)
    cutorch.withDevice(GPU_ID, function() model_2.pnet.rpn = model.pnet.rpn:clone() end)
  end 
--]]
  return model, weights, gradient, training_stats
end

function graph_training(cfg, model_path, snapshot_prefix, training_data_filename, network_filename, network_filename2)
  -- create/load model
  local model, weights, gradient, training_stats = load_model(cfg, model_path, network_filename, 1)
  local model_2= {pnet={}}
  if cfg.finetune == 3 then
    if network_filename2 then
      local model_2_b = load_model(cfg, model_path, network_filename2, 1)
      cutorch.withDevice(GPU_ID, function() model_2.pnet.red_net = model_2_b.pnet.red_net:clone() end)
      cutorch.withDevice(GPU_ID, function() model_2.pnet.enc = model_2_b.pnet.enc:clone() end)
      cutorch.withDevice(GPU_ID, function() model_2.pnet.rpn = model_2_b.pnet.rpn:clone() end)
      model_2_b = nil
    else
      cutorch.withDevice(GPU_ID, function() model_2.pnet.red_net = model.pnet.red_net:clone() end)
      cutorch.withDevice(GPU_ID, function() model_2.pnet.enc = model.pnet.enc:clone() end)
      cutorch.withDevice(GPU_ID, function() model_2.pnet.rpn = model.pnet.rpn:clone() end)
    end
  end

  -- init training status
  if not training_stats then
    training_stats = { pcls={}, preg={}, dcls={}, dreg={} }
  end
  


  print('Reading training data file \'' .. training_data_filename .. '\'.')
  local training_data = torch.load(training_data_filename) 
  training_data = training_data.training_data
  local file_names = keys(training_data.ground_truth) -- in utilities.lua
  print(string.format("Training data loaded. Dataset: '%s'; Total files: %d;)", 
      training_data.dataset_name, 
      #file_names))
  

  local batch_iterator = BatchIterator.new(model, training_data)
  local eval_objective_grad = create_objective(model, weights, gradient, batch_iterator, training_stats, model_2) -- in objective.lua
  
  local rmsprop_state = { learningRate = opt.lr, alpha = opt.rms_decay }
  --local nag_state = { learningRate = opt.lr, weightDecay = 0, momentum = opt.rms_decay }
  local sgd_state = { learningRate = opt.lr, weightDecay = 0.0005, momentum = 0.9 }


  for i=1,500000 do
    --if i % 30000 == 0 then
    if i % 3000 == 0 then
      opt.lr = opt.lr / 2
      rmsprop_state.lr = opt.lr
    end
  
    local timer = torch.Timer()
    local _, loss = optim.rmsprop(eval_objective_grad, weights, rmsprop_state)
    --local _, loss = optim.nag(eval_objective_grad, weights, nag_state)
    --local _, loss = optim.sgd(eval_objective_grad, weights, sgd_state)
    collectgarbage()
    --os.exit()

    local time = timer:time().real

    print(string.format('%d: loss: %f\n', i, loss[1]))
    
    if i%opt.plot == 0 then
      if cfg.flow == 1 then
        plot_training_progress('plot/' .. snapshot_prefix ..'_flow_' .. opt.save_name, training_stats)
      elseif cfg.flow == 2 then
        plot_training_progress('plot/' .. snapshot_prefix ..'_ts_' .. opt.save_name, training_stats)
      else
        plot_training_progress('plot/' .. snapshot_prefix ..'_' .. opt.save_name, training_stats)
      end
    end
    
    if i%opt.snapshot == 0 or i == 1 then
      -- save snapshot
      if cfg.flow == 1 then
        save_model(string.format('trained_model/%s_flow_%s_%06d.t7',opt.save_name, snapshot_prefix, i), weights, opt, training_stats)
      elseif cfg.flow == 2 then
        save_model(string.format('trained_model/%s_ts_%s_%06d.t7',opt.save_name, snapshot_prefix, i), weights, opt, training_stats)
      else
        save_model(string.format('trained_model/%s_%s_%06d.t7',opt.save_name, snapshot_prefix, i), weights, opt, training_stats)
      end
    end

    if i%1000 == 0 then
        if cfg.finetune ~= 0 then
        print('total pos precision: ' .. Det_Table:sum()/Pos_Prop_Table:sum() .. ' ; total neg precision: ' .. Bg_Table:sum()/Neg_Prop_Table:sum())
        print(Pos_Prop_Table)
        print(torch.cdiv(Det_Table, Pos_Prop_Table))
        print(torch.cdiv(Bg_Table, Neg_Prop_Table))
        ----[[
        Det_Table:fill(0)
        Bg_Table:fill(0)
        Pos_Prop_Table:fill(0)
        Neg_Prop_Table:fill(0)
        --]]
      end
    end
    
  end
  -- compute positive anchors, add anchors to ground-truth file
end


function model_validation(cfg, model_path, test_list, test_data_path, network_filename, validate_name)
  local path_name = 'result/' .. validate_name
  local tmp_f, tmp_err = io.open(path_name)
  if tmp_err then
    os.execute('mkdir ' .. path_name)
  end
	-- load model
	local model = load_model(cfg, model_path, network_filename, 0)

	
	-- init detector
	local detector = Detector.new(model)
	
	-- do validation 
	-- video list loop
	local file_count = 1
  local batch_num = cfg.batch_num
  local batch_id = cfg.batch_id
  local total_file = 1000
  local offset = 1
  local file_st = math.floor(total_file/batch_num) * (batch_id-1) +1+offset
  --local file_st = 489
  local file_et = math.floor(total_file/batch_num) * batch_id+offset

  local total_time = 0
  local fea_load_time = 0
  local det_time = 0;
  local in_link_time = 0;
  local between_link_time = 0;
  local st_all = os.clock()
  local total_f = 0
	for video_name in io.lines(test_list) do
    local video_result = nil

    if  math.fmod(file_count,3) == 0 and file_count >= file_st  then
    --if  math.fmod(file_count,3) == 0 and file_count >= 750  then
      --if file_count >= 262 then
  		print('File # ' .. file_count .. ' # ' .. video_name .. ' ##\n')
  		local video_path = test_data_path .. video_name .. '/'
  		local annot_file = video_path .. 'annot.mat'
  		local annot = matio.load(annot_file)
  		annot = annot.annot
  		annot.type(torch.IntTensor)

  		local img_size = annot[1][{{2,3}}]
  		local total_frms = annot:size(1)
      total_f = total_f+total_frms;
  		local start_frm_id = 1 
  		local last_frm_id = start_frm_id + ConvGRU_rho - 1

      local path_buff, score_record = {}, {}
      local top_n = cfg.top_n
  		
  		-- video loop, get feature
  		while(start_frm_id <= total_frms) do
  			local seq_fea = {}
        local st_1 = os.clock()
  			for i = start_frm_id, last_frm_id do
  				if i <= total_frms then
  					local fea_path = video_path .. i .. '.mat'
            if cfg.flow == 1 then
              fea_path = video_path .. 'flow_img/' .. i .. '_flow.mat'
            end
  					local conv_fea = matio.load(fea_path)
  					conv_fea = conv_fea.conv_fea
  					conv_fea.type(torch.Tensor)
  					conv_fea = conv_fea:cuda()
            if cfg.flow == 2 then
              fea_path = video_path .. 'flow_img/' .. i .. '_flow.mat'
              local flow_fea = matio.load(fea_path)
              flow_fea = flow_fea.conv_fea
              flow_fea.type(torch.Tensor)
              flow_fea = flow_fea:cuda()
              conv_fea = func_fea_overlay(conv_fea, flow_fea)
            end
  					table.insert(seq_fea, conv_fea)
  				else
  					table.insert(seq_fea, torch.Tensor(seq_fea[1]:size(1), seq_fea[1]:size(2), seq_fea[1]:size(3)):fill(0):cuda())
  				end	
  			end
        

  			local seq_fea_tensor = func_table_2_tensor(seq_fea)  -- Hjj_utilities
  			local et_1 = os.clock()
        fea_load_time = fea_load_time+et_1-st_1
        --print('feature load time = ' .. et_1 - st_1)
  			-- run detection
        --print('# frame ' .. start_frm_id .. ' - ' .. last_frm_id .. '\n')
        local det_st = os.clock()
  			local result = detector:old_detect(seq_fea_tensor, img_size:clone(), ConvGRU_rho, start_frm_id)
        video_result = func_collect_frm_result(video_result, result, start_frm_id) -- in Hjj_utilities, video_result is tensor
        --print('go inside')
        --local result, temp_in_link = detector:detect(seq_fea_tensor, img_size, ConvGRU_rho, start_frm_id)
        --in_link_time = in_link_time + temp_in_link
        local det_et = os.clock()
        det_time = det_time + det_et - det_st 
        --path_buff, score_record = func_tube_window_mapping(path_buff, result, top_n, start_frm_id) 
        --local path_et = os.clock()
        --between_link_time = between_link_time+path_et-det_et
        --print('now buff size: ' .. start_frm_id .. ' - ' .. #path_buff)   
        --print(path_buff)
        
        --print('detection runtime: ' .. det_ed-det_st .. '\n')
  			start_frm_id = last_frm_id
  			last_frm_id = start_frm_id + ConvGRU_rho -1
  		end -- while(start_frm_id <= total_frms) do

      --[[
      print(#path_buff .. ' -- ' .. top_n)
      print(score_record)
      if #path_buff > top_n then
        local score_record_tensor = torch.Tensor(score_record)
        local s,idx = torch.sort(score_record_tensor)
        local new_table = {}
        --for i=#path_buff-top_n+1,#path_buff do
        for i=1,#path_buff do
          table.insert(new_table, path_buff[idx[i] ])
        end
        --print('#buff3: ' .. #new_table)
        path_buff = new_table
      end
      --]]

      if false then --  debug proposal classification 
        local matio = require 'matio'
         print('##save path ' .. #path_buff)
        for i, v in ipairs(path_buff) do
          local candidate = torch.Tensor(#v.nodes, 15):fill(0)
          for j=1,#v.nodes do
            candidate[j][1] = v.nodes[j].r.minY
            candidate[j][2] = v.nodes[j].r.minX
            candidate[j][3] = v.nodes[j].r.maxY
            candidate[j][4] = v.nodes[j].r.maxX
            candidate[j][5] = v.score
            candidate[j][6] = v.nodes[j].frm
            candidate[j][7] = #path_buff
            candidate[j][8] = v.nodes[j].r2.minY
            candidate[j][9] = v.nodes[j].r2.minX
            candidate[j][10] = v.nodes[j].r2.maxY
            candidate[j][11] = v.nodes[j].r2.maxX
            candidate[j][12] = v.nodes[j].confidence
            candidate[j][13] = v.nodes[j].class
            candidate[j][14] = v.nodes[j].confidence2
            candidate[j][15] = v.nodes[j].class2
          end
          local save_name = path_name .. video_name .. 'path_' .. i .. '.mat' 
          matio.save(save_name,{can = candidate})
        end
      end

      --os.exit()
      local file_name = path_name .. video_name .. '.mat'
      matio.save(file_name, {result = video_result})
      
		end --if  math.fmod(file_count,3) == 0 and file_count >= 1  then
		file_count = file_count + 1

    if file_count > file_et then
      break
    end

	end -- video list loop
   local et_total = os.clock()
   total_time = et_total - st_all

   print(total_time .. '/' .. fea_load_time .. '/' .. det_time .. '/' .. in_link_time .. '/' .. between_link_time .. '/' .. total_f)
  
end

function detect_ucf101(cfg, rpn_cfg_path, fastrcnn_cfg_path, test_list, test_data_path, rpn_filename, fastrcnn_filename, validate_name)
  --local path_name = 'result/' .. validate_name
  local path_name = 'det_bbx/' .. validate_name
  local tmp_f, tmp_err = io.open(path_name)
  if tmp_err then
    os.execute('mkdir ' .. path_name)
  end
  -- load model
  local rpn_model = load_model(cfg, rpn_cfg_path, rpn_filename, -1)
  local fastrcnn_model = load_model(cfg, fastrcnn_cfg_path, fastrcnn_filename, 0)
  
  -- init detector
  local detector = Detector.new(rpn_model, fastrcnn_model)

  -- do validation 
  -- video list loop
  local detect_type = 0 -- 0 single frames; 1 snippet
  local file_count = 1
  local batch_num = cfg.batch_num
  local batch_id = cfg.batch_id
  local total_file = 300
  local offset = 0
  local file_st = math.floor(total_file/batch_num) * (batch_id-1) +1+offset
  --local file_st = 489
  local file_et = math.floor(total_file/batch_num) * batch_id+offset

  local total_time = 0
  local fea_load_time = 0
  local det_time = 0;
  local in_link_time = 0;
  local between_link_time = 0;
  local st_all = os.clock()
  local total_f = 0
  for video_name in io.lines(test_list) do
    local video_result = nil

    if  math.fmod(file_count,1) == 0 and file_count >= file_st  then

      print('File # ' .. file_count .. ' # ' .. video_name .. ' ##\n')
      local video_path = test_data_path .. video_name .. '/'
      local annot_file = video_path .. 'annot.mat'
      local annot = matio.load(annot_file)
      annot = annot.annot
      annot.type(torch.IntTensor)

      local img_size = annot[1][{{2,3}}]
      local total_frms = annot:size(1)
      total_f = total_f+total_frms;
      local start_frm_id = 1 
      local last_frm_id = start_frm_id + ConvGRU_rho - 1

      local path_buff, score_record = {}, {}
      local top_n = cfg.top_n
      
      -- video loop, get feature
      while(start_frm_id <= total_frms) do
        local seq_fea = {}
        local st_1 = os.clock()
        for i = start_frm_id, last_frm_id do
          if i <= total_frms then
            local fea_path = video_path .. i .. '.mat'
            if cfg.flow == 1 then
              fea_path = video_path .. 'flow_img/' .. i .. '_flow.mat'
            end
            local conv_fea = matio.load(fea_path)
            conv_fea = conv_fea.conv_fea
            conv_fea.type(torch.Tensor)
            conv_fea = conv_fea:cuda()
            if cfg.flow == 2 then
              fea_path = video_path .. 'flow_img/' .. i .. '_flow.mat'
              local flow_fea = matio.load(fea_path)
              flow_fea = flow_fea.conv_fea
              flow_fea.type(torch.Tensor)
              flow_fea = flow_fea:cuda()
              conv_fea = func_fea_overlay(conv_fea, flow_fea)
            end
            table.insert(seq_fea, conv_fea)
          else
            --table.insert(seq_fea, torch.Tensor(seq_fea[1]:size(1), seq_fea[1]:size(2), seq_fea[1]:size(3)):fill(0):cuda())
            table.insert(seq_fea, seq_fea[i - start_frm_id])
          end 
        end
        

        local seq_fea_tensor = func_table_2_tensor(seq_fea)  -- Hjj_utilities

        --seq_fea_tensor = func_get_seq_fea('1.t7',cfg.examples_base_path)

        local et_1 = os.clock()
        fea_load_time = fea_load_time+et_1-st_1
        --print('feature load time = ' .. et_1 - st_1)
        -- run detection
        --print('# frame ' .. start_frm_id .. ' - ' .. last_frm_id .. '\n')
        local det_st = os.clock()
        if detect_type == 0 then
          local result, resize_scale = detector:single_detect(seq_fea_tensor, img_size:clone(), ConvGRU_rho, start_frm_id)
          --video_result = func_hmdb_collect_frm_result(video_result, result, start_frm_id,resize_scale) -- in Hjj_utilities, video_result is tensor
          --video_result = func_collect_frm_result(video_result, result, start_frm_id) -- in Hjj_utilities, video_result is tensor

          ----[[
          for f = start_frm_id,last_frm_id do
            if f <= total_frms then
              local video_save_path = string.format('%s%s', path_name, video_name)
              local tmp_f, tmp_err = io.open(video_save_path)
              if tmp_err then
                os.execute('mkdir ' .. video_save_path)
              end
              local frm_det_name = string.format('%s%s/%0.5d.mat', path_name, video_name, f)
              
              local frm_det = result[f - start_frm_id + 1]
              local bbx_num = #frm_det
              local prop = torch.Tensor(bbx_num, 4):fill(0)
              local prob = torch.Tensor(bbx_num, 1):fill(0)
              local loc = torch.Tensor(bbx_num, 4*(cfg.class_count + 1)):fill(0)
              local scores = torch.Tensor(bbx_num, cfg.class_count + 1):fill(0)
              for n = 1, bbx_num do
                loc[n] = frm_det[n].bbx/resize_scale
                scores[n] = frm_det[n].softmax:float()
                prop[n] = frm_det[n].r:totensor()/resize_scale
                prob[n] = frm_det[n].p
              end

              matio.save(frm_det_name,{loc = loc, scores = scores, prop = prop, prob = prob})
            else
              break
            end
          end
          --]]
          
        else
          local result, temp_in_link = detector:snippet_detect(seq_fea_tensor, img_size:clone(), ConvGRU_rho, start_frm_id)
          --in_link_time = in_link_time + temp_in_link
          --local det_et = os.clock()
          --det_time = det_time + det_et - det_st 
          path_buff, score_record = func_tube_window_mapping(path_buff, result, top_n, start_frm_id) 
          --local path_et = os.clock()
          --between_link_time = between_link_time+path_et-det_et
          --print('now buff size: ' .. start_frm_id .. ' - ' .. #path_buff)   
          --print(path_buff)
        end
        --print('detection runtime: ' .. det_ed-det_st .. '\n')
        start_frm_id = last_frm_id+1
        last_frm_id = start_frm_id + ConvGRU_rho -1
      end -- while(start_frm_id <= total_frms) do

      --[[
      print(#path_buff .. ' -- ' .. top_n)
      print(score_record)
      if #path_buff > top_n then
        local score_record_tensor = torch.Tensor(score_record)
        local s,idx = torch.sort(score_record_tensor)
        local new_table = {}
        --for i=#path_buff-top_n+1,#path_buff do
        for i=1,#path_buff do
          table.insert(new_table, path_buff[idx[i] ])
        end
        --print('#buff3: ' .. #new_table)
        path_buff = new_table
      end
      --]]

      if detect_type == 1 then --  debug proposal classification 
        local matio = require 'matio'
         print('##save path ' .. #path_buff)
        for i, v in ipairs(path_buff) do
          local candidate = torch.Tensor(#v.nodes, 15):fill(0)
          for j=1,#v.nodes do
            candidate[j][1] = v.nodes[j].r.minY
            candidate[j][2] = v.nodes[j].r.minX
            candidate[j][3] = v.nodes[j].r.maxY
            candidate[j][4] = v.nodes[j].r.maxX
            candidate[j][5] = v.score
            candidate[j][6] = v.nodes[j].frm
            candidate[j][7] = #path_buff
            candidate[j][8] = v.nodes[j].r2.minY
            candidate[j][9] = v.nodes[j].r2.minX
            candidate[j][10] = v.nodes[j].r2.maxY
            candidate[j][11] = v.nodes[j].r2.maxX
            candidate[j][12] = v.nodes[j].confidence
            candidate[j][13] = v.nodes[j].class
            candidate[j][14] = v.nodes[j].confidence2
            candidate[j][15] = v.nodes[j].class2
          end
          local save_name = path_name .. video_name .. 'path_' .. i .. '.mat' 
          matio.save(save_name,{can = candidate})
        end
      else
        --os.exit()
        --[[
        local file_name = path_name .. video_name .. '.mat'
        matio.save(file_name, {result = video_result})
        --]]
      end
      
    end --if  math.fmod(file_count,3) == 0 and file_count >= 1  then
    file_count = file_count + 1

    if file_count > file_et then
      break
    end

  end -- video list loop
   local et_total = os.clock()
   total_time = et_total - st_all

   print(total_time .. '/' .. fea_load_time .. '/' .. det_time .. '/' .. in_link_time .. '/' .. between_link_time .. '/' .. total_f)
end


function model_test_ucf_sports(cfg, model_path, test_list, test_data_path, network_filename, validate_name)
  local path_name = './result_sports/' .. validate_name
  local tmp_f, tmp_err = io.open(path_name)
  if tmp_err then
    os.execute('mkdir ' .. path_name)
  end
  -- load model
  local model = load_model(cfg, model_path, network_filename, 0)

  
  -- init detector
  local detector = Detector.new(model)
  
  -- do validation 
  -- video list loop
  local file_count = 1
  local batch_num = cfg.batch_num
  local batch_id = cfg.batch_id
  local total_file = 1000
  local offset = 0
  local file_st = math.floor(total_file/batch_num) * (batch_id-1) +1+offset
  --local file_st = 489
  local file_et = math.floor(total_file/batch_num) * batch_id+offset

  
  for video_name in io.lines(test_list) do
    local video_result = nil

    if  math.fmod(file_count,1) == 0 and file_count >= file_st  then
    --if  math.fmod(file_count,3) == 0 and file_count >= 750  then
      --if file_count >= 262 then
      print('File # ' .. file_count .. ' # ' .. video_name .. ' ##\n')
      local video_path = test_data_path .. video_name .. '/im/'

      local path_name = './result_sports/' .. validate_name .. video_name
      local tmp_f, tmp_err = io.open(path_name)
      if tmp_err then
        os.execute('mkdir -p ' .. path_name)
      end


      local annot_file = video_path .. 'annot.mat'
      local annot = matio.load(annot_file)
      annot = annot.annot
      annot.type(torch.IntTensor)

      local img_size = {404, 720};
      local total_frms = annot[1][1]
      local start_frm_id = 1 
      local last_frm_id = start_frm_id + ConvGRU_rho - 1

      local path_buff, score_record = {}, {}
      local top_n = cfg.top_n
      
      -- video loop, get feature
      while(start_frm_id <= total_frms) do
        local seq_fea = {}
        local st_1 = os.clock()
        for i = start_frm_id, last_frm_id do
          if i <= total_frms then
            local fea_path = video_path .. i .. '.mat'
            if cfg.flow == 1 then
              fea_path = video_path .. 'of/' .. i .. '.mat'
            end
            local conv_fea = matio.load(fea_path)
            conv_fea = conv_fea.conv_fea
            conv_fea.type(torch.Tensor)
            conv_fea = conv_fea:cuda()
            if cfg.flow == 2 then
              fea_path = video_path .. 'of/' .. i .. 'mat'
              local flow_fea = matio.load(fea_path)
              flow_fea = flow_fea.conv_fea
              flow_fea.type(torch.Tensor)
              flow_fea = flow_fea:cuda()
              conv_fea = func_fea_overlay(conv_fea, flow_fea)
            end
            table.insert(seq_fea, conv_fea)
          else
            table.insert(seq_fea, torch.Tensor(seq_fea[1]:size(1), seq_fea[1]:size(2), seq_fea[1]:size(3)):fill(0):cuda())
          end 
        end
        

        local seq_fea_tensor = func_table_2_tensor(seq_fea)  -- Hjj_utilities
        local et_1 = os.clock()
        --print('feature load time = ' .. et_1 - st_1)
        -- run detection
        --print('# frame ' .. start_frm_id .. ' - ' .. last_frm_id .. '\n')
        local det_st = os.clock()
        local result = detector:old_detect(seq_fea_tensor, img_size:clone(), ConvGRU_rho, start_frm_id)
        video_result = func_collect_frm_result(video_result, result, start_frm_id) -- in Hjj_utilities, video_result is tensor
        --print('go inside')
        --local result = detector:detect(seq_fea_tensor, img_size, ConvGRU_rho, start_frm_id)
        --path_buff, score_record = func_tube_window_mapping(path_buff, result, top_n, start_frm_id) 
        --print('now buff size: ' .. start_frm_id .. ' - ' .. #path_buff)   
        --print(path_buff)
        

        local det_ed = os.clock()
        --print('detection runtime: ' .. det_ed-det_st .. '\n')
        start_frm_id = last_frm_id
        last_frm_id = start_frm_id + ConvGRU_rho -1
      end -- while(start_frm_id <= total_frms) do

      --[[
      print(#path_buff .. ' -- ' .. top_n)
      print(score_record)
      if #path_buff > top_n then
        local score_record_tensor = torch.Tensor(score_record)
        local s,idx = torch.sort(score_record_tensor)
        local new_table = {}
        --for i=#path_buff-top_n+1,#path_buff do
        for i=1,#path_buff do
          table.insert(new_table, path_buff[idx[i] ])
        end
        --print('#buff3: ' .. #new_table)
        path_buff = new_table
      end
      --]]

      if false then --  debug proposal classification 
        local matio = require 'matio'
         print('##save path ' .. #path_buff)
        for i, v in ipairs(path_buff) do
          local candidate = torch.Tensor(#v.nodes, 15):fill(0)
          for j=1,#v.nodes do
            candidate[j][1] = v.nodes[j].r.minY
            candidate[j][2] = v.nodes[j].r.minX
            candidate[j][3] = v.nodes[j].r.maxY
            candidate[j][4] = v.nodes[j].r.maxX
            candidate[j][5] = v.score
            candidate[j][6] = v.nodes[j].frm
            candidate[j][7] = #path_buff
            candidate[j][8] = v.nodes[j].r2.minY
            candidate[j][9] = v.nodes[j].r2.minX
            candidate[j][10] = v.nodes[j].r2.maxY
            candidate[j][11] = v.nodes[j].r2.maxX
            candidate[j][12] = v.nodes[j].confidence
            candidate[j][13] = v.nodes[j].class
            candidate[j][14] = v.nodes[j].confidence2
            candidate[j][15] = v.nodes[j].class2
          end
          local save_name = path_name .. video_name .. '/path_' .. i .. '.mat' 
          matio.save(save_name,{can = candidate})
        end
      end
      local file_name = path_name .. video_name .. 'result.mat'
      matio.save(file_name, {result = video_result})
      
    end --if  math.fmod(file_count,3) == 0 and file_count >= 1  then
    file_count = file_count + 1

    if file_count > file_et then
      break
    end

  end -- video list loop
end

--model_test_ucf_sports(cfg, opt.model, opt.ucf_sp_test_list, opt.ucf_sp_test_data, opt.restore, opt.validate_name)

--graph_training(cfg, opt.model, opt.name, opt.train , opt.restore, opt.restore2)
--graph_training(cfg, opt.model, opt.name, opt.train , nil, nil)

--model_validation(cfg, opt.model, opt.test_list, opt.test_data, opt.restore, opt.validate_name)

detect_ucf101(cfg, opt.model,opt.model, opt.test_list, opt.test_data, opt.restore,opt.restore2, opt.validate_name)

  