require 'cunn'
require 'cutorch'
require 'Hjj_Localizer'
require 'Hjj_encoder_decoder_ConvGRU'
require 'Anchors'
require 'Hjj_utilities'
require 'Hjj_global'
require 'Rect'
require 'nms'
require 'Hjj_linking'

local Detector = torch.class('Detector')

function Detector:__init(model, fastrcnn_model)
  local cfg = model.cfg
  self.nms_threshold = cfg.nms_threshold
  self.iou_threshold = cfg.iou_threshold
  self.model = model
  self.anchors = Anchors.new(model.pnet.rpn, model.cfg.scales)
  self.localizer = Localizer.new(model.pnet.rpn.outnode.children[5])
  if fastrcnn_model then
	self.fastrcnn_model = fastrcnn_model
  	self.fastrcnn_anchors = Anchors.new(fastrcnn_model.pnet.rpn, fastrcnn_model.cfg.scales)
  	self.fastrcnn_localizer = Localizer.new(fastrcnn_model.pnet.rpn.outnode.children[5])
  end
  self.localizer2 = Localizer.new()
  self.lsm = nn.SoftMax():cuda() -- for training it is logsoftmax
  self.amp = nn.SpatialAdaptiveMaxPooling(cfg.roi_pooling.kw, cfg.roi_pooling.kh):cuda()
end

function Detector:single_detect(input, origin_img_size, seq_len, start_frm_id)
  local nms_threshold = self.nms_threshold
  local iou_threshold = self.iou_threshold
  local cfg = self.model.cfg
  --rpn
  local rpn_red_net = self.model.pnet.red_net
  rpn_red_net:evaluate()
  local rpn_enc = self.model.pnet.enc
  rpn_enc:evaluate()
  local rpn_rpn = self.model.pnet.rpn
  rpn_rpn:evaluate()
  local rpn_anchors = self.anchors

  -- fastrcnn
  local fastrcnn_red_net = self.fastrcnn_model.pnet.red_net
  fastrcnn_red_net:evaluate()
  local fastrcnn_enc = self.fastrcnn_model.pnet.enc
  fastrcnn_enc:evaluate()
  local fastrcnn_rpn = self.fastrcnn_model.pnet.rpn
  fastrcnn_rpn:evaluate()
  local fastrcnn_cnet = self.fastrcnn_model.cnet
  --fastrcnn_cnet:training()
  fastrcnn_cnet:evaluate()
  local fastrcnn_localizer = self.fastrcnn_localizer

  local kh, kw = cfg.roi_pooling.kh, cfg.roi_pooling.kw
  local bgclass = cfg.class_count + 1   -- background class
  local amp = self.amp
  local lsm = self.lsm
  local cnet_input_planes = self.fastrcnn_model.layers[#self.fastrcnn_model.layers].filters
  
  local img_size, resize_scale = func_processRoIs(cfg, origin_img_size)
  local input_rect = Rect.new(0, 0, img_size[2], img_size[1])
  
  -- pass image through rpn
  local st_1 = os.clock()
  local rpn_red_output = rpn_red_net:forward(input)
  local rpn_enc_output = rpn_enc:forward(rpn_red_output[1])
  forwardConnection(rpn_enc,rpn_rpn)
  local rpn_outputs = rpn_rpn:forward(rpn_red_output[2])

  local fastrcnn_red_output = fastrcnn_red_net:forward(input)
  local fastrcnn_enc_output = fastrcnn_enc:forward(fastrcnn_red_output[1])
  forwardConnection(fastrcnn_enc,fastrcnn_rpn)
  local fastrcnn_outputs = fastrcnn_rpn:forward(fastrcnn_red_output[2])
  local et_1 = os.clock()
  --print('forward time : ' .. et_1 - st_1)

  local outputs = rpn_outputs
  -- for each time stamp
  local detection_results = {}
  for t=1, seq_len do
	local matches = {}
	local st_1 = os.clock()
	local ccc = 0
	local aspect_ratios = 3
	for i=1,4 do 
		local layer = outputs[i][t]
		local layer_size = layer:size()

		for a=1,aspect_ratios do
			local ofs = (a-1) * 6
			local cls_out = layer[{{ofs + 1, ofs + 2}, {}, {}}] 
			local reg_out = layer[{{ofs + 3, ofs + 6}, {}, {}}]
			local soft_cls = lsm:forward(cls_out)
			local bool_mat = torch.gt(soft_cls[{{1},{},{}}],iou_threshold)
			--print(bool_mat:size())
			local idx = torch.nonzero(bool_mat)
			--print('idx -- ' .. i .. ' -- a -- ' .. a)
			--print(idx)
			if idx:size():size() ~= 0 then
				--print(idx:size())
				for j=1,idx:size(1) do
					local anc = rpn_anchors:get(i,a,idx[j][2],idx[j][3])
					--print(reg_out[{{},{idx[j][2]}, {idx[j][3]}}])
					local r
					--local r = Anchors.anchorToInput(anc, {0,0,0,0})
					if cfg.enable_reg == 1 then
						r = Anchors.anchorToInput(anc, reg_out[{{},{idx[j][2]}, {idx[j][3]}}]:squeeze())
					else
					 	r = Anchors.anchorToInput(anc, {0,0,0,0})
					 end
					if r:overlaps(input_rect)  then
						  r=r:clip(input_rect)
						  table.insert(matches, { p=soft_cls[1][idx[j][2]][idx[j][3]], a=anc, r=r, l=i })
					end
				end
			end
		end

	end
	
	local et_1 = os.clock()
  	--print(#matches .. 'rpn time : ' .. et_1 - st_1 .. ' ;time = ' .. ccc)


	local winners = {}
	
	if #matches > 0 then
		-- NON-MAXIMUM SUPPRESSION
		local bb = torch.Tensor(#matches, 4)
		local score = torch.Tensor(#matches, 1)
		for i=1,#matches do
		  bb[i] = matches[i].r:totensor()
		  score[i] = matches[i].p
		end
		
		
		local pick = nms(bb, nms_threshold, score)
		local candidates = {}
		pick:apply(function (x) table.insert(candidates, matches[x]) end )
		--print(string.format('candidates: %d', #candidates))
		--print(candidates)
		
		-- REGION CLASSIFICATION 
		-- create cnet input batch
		local cinput = torch.CudaTensor(#candidates, cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes)
		-- RoI pooling

		for i,v in ipairs(candidates) do
		  -- pass through adaptive max pooling operation
		  local pi, idx = extract_roi_pooling_input(v.r, fastrcnn_localizer, fastrcnn_outputs[5][t])
		  cinput[i] = amp:forward(pi):view(cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes)
		  --print(v.r)
		end
		
		local coutputs=fastrcnn_cnet:forward({cinput})
		local bbox_out = coutputs[1][1]:clone()
		local cls_out = coutputs[2][1]:clone()

		local yclass = {}

		if false then --  debug proposal classification 
			local matio = require 'matio'
			local candidate = torch.Tensor(#candidates, 6):fill(0)
			for i,v in ipairs(candidates) do 
				candidate[i][1] = v.r.minY
				candidate[i][2] = v.r.minX
				candidate[i][3] = v.r.maxY
				candidate[i][4] = v.r.maxX
				candidate[i][5] = v.p
				local cprob = cls_out[i]
		  		--print(torch.type(cprob))
		  		local p,c = torch.sort(cprob, 1, true) -- get probabilities and class indicies
				candidate[i][6] = c[1]
			end
			local save_name = './detect/candidates_' .. start_frm_id-1+t .. '.mat' 
			matio.save(save_name,{can = candidate})
		end
		--print(cls_out)
		for i,x in ipairs(candidates) do
		  local cprob = cls_out[i]
		  --print(cprob)
		  local p,c = torch.sort(cprob, 1, true) -- get probabilities and class indicies
		  --print(c)
		  x.class = c[1]
		  x.confidence = p[1]
		  x.softmax = cprob:clone()
		  --print(bbox_out:size())
		  --print(bbox_out[{ {i},{1+(x.class-1)*4, x.class*4} }])
		  x.r2 = Anchors.anchorToInput(x.r, bbox_out[{ {i},{1+(x.class-1)*4, x.class*4} }]:squeeze())

		  x.bbx = torch.Tensor(1,4*bgclass):fill(0)
		  for cls_n=1,bgclass do
		  	local r = Anchors.anchorToInput(x.r, bbox_out[{ {i},{1+(cls_n-1)*4, cls_n*4} }]:squeeze())
		  	r=r:clip(input_rect)
		  	x.bbx[{{1},{1+(cls_n-1)*4}}] = r.minY
		  	x.bbx[{{1},{2+(cls_n-1)*4}}] = r.minX
		  	x.bbx[{{1},{3+(cls_n-1)*4}}] = r.maxY
		  	x.bbx[{{1},{4+(cls_n-1)*4}}] = r.maxX
		  end

		  --[[
		  if true then
			if not yclass[x.class] then
			  yclass[x.class] = {}
			end
			table.insert(yclass[x.class], x)
		  end
		--]]
		table.insert(winners, x)
		end

		
		-- run per class NMS
		--[[
		for i,c in pairs(yclass) do
		  -- fill rect tensor
		  bb = torch.Tensor(#c, 5)
		  for j,r in ipairs(c) do
			bb[{j, {1,4}}] = r.r2:totensor()
			bb[{j, 5}] = r.confidence
		  end
		  
		  pick = nms(bb, 1, bb[{{}, {5}}])
		  pick:apply(function (x) table.insert(winners, c[x]) end ) 
		end	
		--]]
	end -- if #matches > 0
	table.insert(detection_results, winners)
  end -- for t=1,...
  --os.exit()
  return detection_results, resize_scale
end




function Detector:detect(input, origin_img_size, seq_len, start_frm_id)
  local chain = {}
  local nms_threshold = self.nms_threshold
  local iou_threshold = self.iou_threshold
  local cfg = self.model.cfg
  local red_net = self.model.pnet.red_net
  red_net:evaluate()
  local enc = self.model.pnet.enc
  enc:evaluate()
  local rpn = self.model.pnet.rpn
  rpn:evaluate()
  local cnet = self.model.cnet
  cnet:evaluate()
  local kh, kw = cfg.roi_pooling.kh, cfg.roi_pooling.kw
  local bgclass = cfg.class_count + 1   -- background class
  local amp = self.amp
  local lsm = self.lsm
  local cnet_input_planes = self.model.layers[#self.model.layers].filters
  
  local img_size, resize_scale = func_processRoIs(cfg, origin_img_size)
  local input_rect = Rect.new(1, 1, img_size[2], img_size[1])
  
  local function func_get_detected(outputs, t, aspect_ratios, scales, start_id)
  	local matches = {}
  	for i=1,scales do 
		local layer = outputs[i][t]
		local layer_size = layer:size()
		for a=1,aspect_ratios do
			local ofs = (a-1) * 6
			local cls_out = layer[{{ofs + 1, ofs + 2}, {}, {}}] 
			local reg_out = layer[{{ofs + 3, ofs + 6}, {}, {}}]
			local soft_cls = lsm:forward(cls_out)
			local bool_mat = torch.gt(soft_cls[{{1},{},{}}],iou_threshold)
			--print(bool_mat:size())
			local idx = torch.nonzero(bool_mat)
			if idx:size():size() ~= 0 then
				--print(idx:size())
				for j=1,idx:size(1) do
					local anc = self.anchors:get(i,a,idx[j][2],idx[j][3])
					local r 
					if cfg.enable_reg == 1 then
						r = Anchors.anchorToInput(anc, reg_out[{{},{idx[j][2]}, {idx[j][3]}}]:squeeze())
					else
					 	r = Anchors.anchorToInput(anc, {0,0,0,0})
					 end
					if r:overlaps(input_rect)  then
						r=r:clip(input_rect)
						  table.insert(matches, { frm =start_frm_id ,p=soft_cls[1][idx[j][2]][idx[j][3]], a=anc, r=r, l=i })
					end
				end
			end
		end
	end
	--print('#matches: ' .. #matches)
	local winners = {}
	if #matches > 0 then
		-- NON-MAXIMUM SUPPRESSION
		local bb = torch.Tensor(#matches, 4)
		local score = torch.Tensor(#matches, 1)
		for i=1,#matches do
		  bb[i] = matches[i].r:totensor()
		  score[i] = matches[i].p
		end
		
		local pick = nms(bb, nms_threshold, score)
		local candidates = {}
		pick:apply(function (x) table.insert(candidates, matches[x]) end )
		--print(string.format('candidates: %d', #candidates))
		--print(candidates)
		
		-- REGION CLASSIFICATION 
		-- create cnet input batch
		local cinput = torch.CudaTensor(#candidates, cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes)
		-- RoI pooling
		for i,v in ipairs(candidates) do
		  -- pass through adaptive max pooling operation
		  local pi, idx = extract_roi_pooling_input(v.r, self.localizer, outputs[5][t])
		  cinput[i] = amp:forward(pi):view(cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes)
		  -- faster-rcnn single frame feature 
		  local pi, idx = extract_roi_pooling_input(v.r, self.localizer2, input[t])
		  if cfg.flow == 2 then
		  	v.roi_fea  = amp:forward(pi):view(cfg.roi_pooling.kw * cfg.roi_pooling.kh *1024):clone()
		  else
			v.roi_fea  = amp:forward(pi):view(cfg.roi_pooling.kw * cfg.roi_pooling.kh * 512):clone()
		  end
		end
		
		-- send extracted roi-data through classification network
		local coutputs = cnet:forward({cinput}) -- cnet is sequencer
		local bbox_out = coutputs[1][1]
		local cls_out = coutputs[2][1]
		
		local yclass = {}

		if false then --  debug proposal classification 
			local matio = require 'matio'	
			local candidate = torch.Tensor(#candidates, 7):fill(0)
			local fea_can = torch.Tensor(#candidates, 25088):fill(0)
			for i,v in ipairs(candidates) do 
				candidate[i][1] = v.r.minY
				candidate[i][2] = v.r.minX
				candidate[i][3] = v.r.maxY
				candidate[i][4] = v.r.maxX
				candidate[i][5] = v.p
				candidate[i][6] = math.exp(cls_out[i][25])
				candidate[i][7] = i
				fea_can[i] = v.roi_fea:float()
			end
			local save_name = './detect/candidates_' .. start_frm_id-1+t .. '.mat' 
			matio.save(save_name,{can = candidate, roi_fea = fea_can})
		end

		for i,x in ipairs(candidates) do
		  x.r2 = Anchors.anchorToInput(x.r, bbox_out[i])
		  x.r2=x.r2:clip(input_rect)
		  local cprob = cls_out[i]
		  --print(cprob)
		  local p,c = torch.sort(cprob, 1, true) -- get probabilities and class indicies
		  
		  x.class = c[1]
		  x.confidence = p[1]
		  x.softmax = cprob:clone()
		  --print(x.class)
		  if true then
			if not yclass[x.class] then
			  yclass[x.class] = {}
			end

			table.insert(yclass[x.class], x)
		  end
		end
		
		-- run per class NMS
		for i,c in pairs(yclass) do
		  -- fill rect tensor
		  bb = torch.Tensor(#c, 5)
		  for j,r in ipairs(c) do
			bb[{j, {1,4}}] = r.r2:totensor()
			bb[{j, 5}] = r.confidence
		  end
		  
		  pick = nms(bb, nms_threshold, bb[{{}, {5}}])
		  pick:apply(function (x) table.insert(winners, c[x]) end ) 
		end
		
	end -- if #matches > 0
	-- number detected objects
	
	for i, v in ipairs(winners) do
		v.id = start_id+i-1
		v.sub_id = 1
	end
	return winners
	
  end -- function func_get_detected

  local function func_get_nodes(objs)
  	local nodes = {}
  	if #objs == 0 then
  		return nodes
  	end
  	for i,v in ipairs(objs) do
  		if nodes[v.id] == nil then
  			nodes[v.id] = {sub_nodes = {}, id = v.id}
  			table.insert(nodes[v.id].sub_nodes, v)
  		else
  			table.insert(nodes[v.id].sub_nodes, v)
  		end
  	end

  	return nodes
  end

  local function func_get_representative(nodes)
  	local representative = {id=nodes.id, r=nil, roi_fea=nil}
  	if #nodes.sub_nodes == 1 then
  		representative.r = Rect.new(nodes.sub_nodes[1].r.minX, nodes.sub_nodes[1].r.minY,nodes.sub_nodes[1].r.maxX , nodes.sub_nodes[1].r.maxY )
  		representative.roi_fea = nodes.sub_nodes[1].roi_fea
  		return representative
  	end
  	--[[
  	local map = torch.Tensor(800,800):fill(0)
  	for i,v in ipairs(nodes.sub_nodes) do
  		local minY = math.ceil(v.r.minY)
  		local minX = math.ceil(v.r.minX)
		local maxY = math.floor(v.r.maxY)
		local maxX = math.floor(v.r.maxX)
		map[{{minY, maxY}, {minX, maxX}}] = map[{{minY, maxY}, {minX, maxX}}] + 1
  	end
  	local minx,miny,maxx,maxy = func_get_center(map, resize_scale) -- Hjj_utilities
  	representative.r = Rect.new(minx,miny,maxx,maxy)
  	--]]
  	local max_s = 0
  	for i,v in ipairs(nodes.sub_nodes) do
  		if v.p >max_s then
  			max_s = v.p
	  		local minY = math.ceil(v.r.minY)
	  		local minX = math.ceil(v.r.minX)
			local maxY = math.floor(v.r.maxY)
			local maxX = math.floor(v.r.maxX)
			representative.r = Rect.new(minX,minY,maxX,maxY)
			representative.roi_fea = v.roi_fea
		end
  	end
  	return representative
  end









  local total_t = 0
  -- pass image through network
  local red_output = red_net:forward(input)
  local enc_output = enc:forward(red_output[1])
  forwardConnection(enc,rpn)
  local outputs = rpn:forward(red_output[2])
  local aspect_ratios = 3

  -- forward tracking
  local start_id = 1 -- number object trajectories
  local detected_objs = func_get_detected(outputs, 1, aspect_ratios, 4, start_id) --(outputs, t, aspect_ratios, scales, start_id)
  local path_available = torch.Tensor(#detected_objs):fill(0)
  --print('det_obj ' .. #detected_objs)
  for t=2, ConvGRU_rho do
  	-- forward pass
  	local last_frame_nodes = func_get_nodes(detected_objs)
  	local chain_num = #last_frame_nodes
  	--print('chain_num: ' .. chain_num)
  	local representative_table = {}
  	table.insert(chain, last_frame_nodes)
  	local winners = {}
  	for j=1, chain_num do
  		if last_frame_nodes[j] then
  			local matches_in_current_frm = {}
	  		--print('t -- ' .. t .. ' ; j -- ' .. j .. ';')
	  		local representative = func_get_representative(last_frame_nodes[j])
	  		if not representative_table[j] then
	  			representative_table[j] = representative
	  		end
	  		--print(representative)
	  		local det_flag = 0
	  		for s=1,4 do  -- 4 scales
	  			local tmp_localizer = self.anchors.localizers[s]
	  			local rect_fea = tmp_localizer:inputToFeatureRect(representative.r)
	  			local centerX, centerY = rect_fea:center()
	  			centerX = math.max(1,math.floor(centerX))
	  			centerY = math.max(1,math.floor(centerY))
	  			local cur_output = outputs[s][t]
	  			local cur_output_size = cur_output:size()
	  			local x_y_pairs = func_get_nearby_center(centerX,centerY,cur_output_size) -- Hjj_utilities
	  			for k, w in ipairs(x_y_pairs) do
	  				local x=w.x
	  				local y=w.y
	  				--print(cur_output:size())
	  				--print(x .. '  ' .. y)
	  				local c = cur_output[{{}, y, x}]
	  				for a=1, aspect_ratios do
	  					local ofs = (a-1) * 6
						local cls_out = c[{{ofs + 1, ofs + 2}}] 
						local reg_out = c[{{ofs + 3, ofs + 6}}]
						
						-- classification
						local c = lsm:forward(cls_out)
						if c[1] > iou_threshold then
							local anc = self.anchors:get(s,a,y,x)
							-- regression
							local r
							if cfg.enable_reg == 1 then
								r = Anchors.anchorToInput(anc, reg_out)
							else
								r = Anchors.anchorToInput(anc, {0,0,0,0})
							end
							if r:overlaps(input_rect) and r:overlaps(representative.r) then
							  r=r:clip(input_rect)
							  table.insert(matches_in_current_frm, {frm = start_frm_id+t-1, p=c[1], a=anc, r=r, l=s, np=c[2], pos={i=s,a=a,y=y,x=x}, id= representative.id})
								det_flag = 1
							end
						end
					end -- for a=1,aspect_ratios
	  			end -- for k, w in ipairs(x_y_pairs) 
	  		end --for s=1,4 do
	  		if det_flag == 0 then
	  			-- chain ended because the obj faded
	  			path_available[j] = 1
	  		end
	  		if #matches_in_current_frm > 0 then
		  		local bb = torch.Tensor(#matches_in_current_frm, 4)
		  		local score = torch.Tensor(#matches_in_current_frm, 1)
				for j=1,#matches_in_current_frm do
				  bb[j] = matches_in_current_frm[j].r:totensor()
				  score[j] = matches_in_current_frm[j].p
				end

				local pick = nms(bb, nms_threshold, score)
				local tmp_candidates = {}
				local candidates = {}
				pick:apply(function (x) table.insert(tmp_candidates, matches_in_current_frm[x]) end)
				-- REGION CLASSIFICATION 
				-- create cnet input batch
				local cinput = torch.CudaTensor(#tmp_candidates, cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes)

				local thd1,thd2 = 0.35,0.4
				for k,v in ipairs(tmp_candidates) do
				  -- pass through adaptive max pooling operation
				  local pi, idx = extract_roi_pooling_input(v.r, self.localizer, outputs[5][t])
				  cinput[k] = amp:forward(pi):view(cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes)
				  local pi, idx = extract_roi_pooling_input(v.r, self.localizer2, input[t])
				  --print(v.r)
		  		  if cfg.flow == 2 then
				  	v.roi_fea  = amp:forward(pi):view(cfg.roi_pooling.kw * cfg.roi_pooling.kh *1024):clone()
				  else
					v.roi_fea  = amp:forward(pi):view(cfg.roi_pooling.kw * cfg.roi_pooling.kh * 512):clone()
				  end
		  		  --print(v.roi_fea:float():sum())
		  		  table.insert(candidates,v)
		  		  --[[
		  		  if torch.dist(representative.roi_fea, v.roi_fea)/torch.norm(representative.roi_fea) < thd1 and torch.dist(representative_table[j].roi_fea, v.roi_fea)/torch.norm(representative_table[j].roi_fea) < thd2 then
		  		  	table.insert(candidates,v)
		  		  end
		  		  --]]

				end

				--[[
				if #candidates==0 then
					local tmp_thd1, tmp_thd2 = thd1+0.2, thd2+0.2
					for k,v in ipairs(tmp_candidates) do
				  -- pass through adaptive max pooling operation
				  local pi, idx = extract_roi_pooling_input(v.r, self.localizer, outputs[5][t])
				  cinput[k] = amp:forward(pi):view(cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes)
				  local pi, idx = extract_roi_pooling_input(v.r, self.localizer2, input[t])
				  --print(v.r)
		  		  v.roi_fea  = amp:forward(pi):view(cfg.roi_pooling.kw * cfg.roi_pooling.kh * 512):clone()
		  		  --print(v.roi_fea:float():sum())

		  		  if torch.dist(representative.roi_fea, v.roi_fea)/torch.norm(representative.roi_fea) < tmp_thd1 and torch.dist(representative_table[j].roi_fea, v.roi_fea)/torch.norm(representative_table[j].roi_fea) < tmp_thd2 then
		  		  	table.insert(candidates,v)
		  		  end
				end
				end
				--]]

				local coutputs = cnet:forward({cinput}) -- cnet is sequencer
				local bbox_out = coutputs[1][1]
				local cls_out = coutputs[2][1]
				
				local yclass = {}

				if false then --  debug proposal classification 
					local matio = require 'matio'	
					local candidate = torch.Tensor(#candidates, 7):fill(0)
					local fea_can = torch.Tensor(#candidates, 25088):fill(0)
					for i,v in ipairs(candidates) do 
						candidate[i][1] = v.r.minY
						candidate[i][2] = v.r.minX
						candidate[i][3] = v.r.maxY
						candidate[i][4] = v.r.maxX
						candidate[i][5] = v.p
						candidate[i][6] = math.exp(cls_out[i][25])
						candidate[i][7] = v.id
						fea_can[i] = v.roi_fea:float()
						--print(v.roi_fea:float():sum())
					end
					--os.exit()
					local save_name = './detect/candidates_' .. start_frm_id-1+t .. '_' .. j .. '.mat' 
					--print(save_name)
					matio.save(save_name,{can = candidate, roi_fea = fea_can})
				end

				for i,x in ipairs(candidates) do
				  x.r2 = Anchors.anchorToInput(x.r, bbox_out[i])
				  x.r2=x.r2:clip(input_rect)
				  local cprob = cls_out[i]
				  --print(cprob)
				  local p,c = torch.sort(cprob, 1, true) -- get probabilities and class indicies
				  
				  x.class = c[1]
				  x.confidence = p[1]
				  x.softmax = cprob:clone()
				  --print(x.class)
				  --if x.class ~= bgclass then
				  if true then
					if not yclass[x.class] then
					  yclass[x.class] = {}
					end
					table.insert(yclass[x.class], x)
				  end
				end
				
				-- run per class NMS
				for i,c in pairs(yclass) do
				  -- fill rect tensor
				  bb = torch.Tensor(#c, 5)
				  for j,r in ipairs(c) do
					bb[{j, {1,4}}] = r.r2:totensor()
					bb[{j, 5}] = r.confidence
				  end
				  
				  pick = nms(bb, nms_threshold, bb[{{}, {5}}])
				  pick:apply(function (x) table.insert(winners, c[x]) end ) 
				end
		  	end --if #matches_in_current_frm > 0 then
	  		--print('#matches_in_current_frm: '.. #matches_in_current_frm)
	  		representative_table[j] = representative
  		end
  	end -- j=1, chain_num do

  	detected_objs = winners
  	--print('after nms: ' .. #detected_objs)
	if false then --  debug proposal classification 
		local matio = require 'matio'	
		local candidate = torch.Tensor(#detected_objs, 6):fill(0)
		for i,v in ipairs(detected_objs) do 
			candidate[i][1] = v.r.minY
			candidate[i][2] = v.r.minX
			candidate[i][3] = v.r.maxY
			candidate[i][4] = v.r.maxX
			candidate[i][5] = v.p
			candidate[i][6] = v.id
		end
		local save_name = './detect/tracked_' .. start_frm_id-1+t .. '.mat' 
		matio.save(save_name,{can = candidate})
	end

  end -- for t

  table.insert(chain, func_get_nodes(detected_objs))

  local total_t= 0;
  st = os.clock()
  local path =  func_path_planning(chain, path_available) -- Hjj_linking 
  et = os.clock()
  total_t = et-st+total_t
  print('path: ' .. total_t)
  if false then --  debug proposal classification 
	local matio = require 'matio'
	for i, v in ipairs(path) do
		local candidate = torch.Tensor(#v.nodes, 6):fill(0)
		for j=1,#v.nodes do
			candidate[j][1] = v.nodes[j].r.minY
			candidate[j][2] = v.nodes[j].r.minX
			candidate[j][3] = v.nodes[j].r.maxY
			candidate[j][4] = v.nodes[j].r.maxX
			candidate[j][5] = v.score
			candidate[j][6] = v.nodes[j].id
		end
		local save_name = './detect/path_' .. start_frm_id .. '_' .. i .. '.mat' 
		matio.save(save_name,{can = candidate})
	end
  end
  print('path -- ' .. #path)
  --os.exit()
  return path, total_t
end







