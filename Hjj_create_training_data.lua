require 'torch'
require 'cutorch'
require 'Hjj_global' -- ConvGRU_rho
require 'Hjj_utilities'
require 'Rect' 
local matio = require 'matio'

function func_get_roi(gt_tensor)
	-- gt_tensor is 1 dim tensor
	local roi = {}

	local obj_num = gt_tensor[6]
	if obj_num == 0 then
		return roi
	end

	local class_id = gt_tensor[5]
	top_offset = 9
	bottom_offset = top_offset + 4 -1
	while(top_offset < gt_tensor:size(1) and obj_num > 0) do
		if gt_tensor[bottom_offset] + gt_tensor[bottom_offset-1] > 0 then
			local temp_roi = {
				rect = Rect.new(gt_tensor[top_offset], gt_tensor[top_offset+1], gt_tensor[top_offset+2]+gt_tensor[top_offset], gt_tensor[top_offset+3]+gt_tensor[top_offset+1]),
				class_index = class_id
			} -- in gt_tensor , bbx=(minx, miny,w,h)
			obj_num = obj_num - 1
			table.insert(roi, temp_roi)
		end
		top_offset = bottom_offset + 3
		bottom_offset = top_offset + 4 -1
	end

	return roi
end


function func_create_training_data()
	local data_root_dir = '/media/zangxh/JJH/sahasuman-bmvc2016_code-fb0de209d699/hmdb21_train_test_data/'
	local data_dir = data_root_dir .. 'train_test_data/'
	--local video_name_list_file = data_root_dir .. 'train_video_list.txt'
	local video_name_list_file = data_root_dir .. 'new_train_list.txt'
	local save_data_path = './data/hmdb_training_new/'
	local seq_len = ConvGRU_rho -- gloabl variant
	local count = 1
	local training_data = {}
	training_data.dataset_name = 'hmdb'	
	training_data.training_set = {}
	training_data.ground_truth = {}

	local file_count = 1
	for video_name in io.lines(video_name_list_file) do
		print('File # ' .. file_count .. ' # ' .. video_name .. ' ##\n')
		local video_path = data_dir .. video_name .. '/'
		local annot_file = video_path .. 'annot.mat'
		local annot = matio.load(annot_file)
		annot = annot.annot
		annot.type(torch.IntTensor)
		local total_frms = annot:size(1)
		local start_frm_id = 1
		while (annot[start_frm_id][6]==0) do
			start_frm_id = start_frm_id + 1
		end
		local last_frm_id = start_frm_id + seq_len -1

		while last_frm_id <= total_frms do	
			print('\t Seq ' .. count .. ' : ' ..  start_frm_id .. ' to ' .. last_frm_id .. ' :\n')
			local seq_fea = {}
			local seq_ground_truth = {}			
			for i=start_frm_id,last_frm_id do
				local fea_path = video_path .. i .. '.mat'
				--print(fea_path)
				local conv_fea = matio.load(fea_path)
				conv_fea = conv_fea.conv_fea
				conv_fea.type(torch.Tensor)
				conv_fea = conv_fea:cuda()
				table.insert(seq_fea, conv_fea)	

				local img_ground_truth = {}
				img_ground_truth.img_size = annot[i][{{2,3}}]
				img_ground_truth.rois = func_get_roi(annot[i])
				img_ground_truth.info = {vn=video_name, frm=i}

				for j=1,#img_ground_truth.rois do
					print('\t\t Class = ' .. img_ground_truth.rois[j].class_index .. ' ; Rect = [ ' .. img_ground_truth.rois[j].rect.minX .. ', ' .. img_ground_truth.rois[j].rect.minX .. ', ' .. img_ground_truth.rois[j].rect.maxX .. ', ' .. img_ground_truth.rois[j].rect.maxY .. ' ]\n')
				end

				table.insert(seq_ground_truth, img_ground_truth)
			end
			local seq_fea_name = count .. '.t7'
			local seq_fea_path = save_data_path .. seq_fea_name
			local seq_fea_tensor = func_table_2_tensor(seq_fea)
			torch.save(seq_fea_path, {seq_fea = seq_fea_tensor})
			table.insert(training_data.training_set, count .. '.t7')

			training_data.ground_truth[seq_fea_name] = seq_ground_truth -- file name as index

			start_frm_id = start_frm_id + torch.random(torch.Generator(),1,seq_len)
			if start_frm_id < total_frms then
				while (annot[start_frm_id][6]==0 and start_frm_id < total_frms) do
					start_frm_id = start_frm_id + 1
				end
			end
			last_frm_id = start_frm_id + seq_len - 1
			count = count + 1
		end
		file_count = file_count + 1
		--[[
		if file_count > 10 then 
			break
		end
		--]]
	end
	torch.save(save_data_path .. 'ground_truth.t7' , {training_data = training_data})
end

-- generate optical flow training sequences
function func_create_training_flow_data()
	local data_root_dir = '/media/zangxh/JJH/sahasuman-bmvc2016_code-fb0de209d699/ucf_train_test_data/'
	local data_dir = data_root_dir .. 'train_test_data/'

	local save_data_path = './data/flow_training/'
	local seq_len = ConvGRU_rho -- gloabl variant

	local rgb_data = torch.load('data/training/ground_truth.t7') 
  	rgb_data = rgb_data.training_data

  	local id = 6
  	local num = 6

  	local st = math.floor(#rgb_data.training_set/num) * (id-1) + 1
  	--local st = 24789
  	local et = math.floor(#rgb_data.training_set/num) * id
  	if id == num then
  		et = #rgb_data.training_set
  	end

  	for i=st, et do
  		print(id .. ' -- ' .. i .. ' -- ' .. #rgb_data.training_set .. ' : ' .. rgb_data.training_set[i])
  		--local st_all = os.clock()
  		--[[
  		local flow_name = string.gsub(rgb_data.training_set[i],'.t7','_flow.t7')
  		torch.load(save_data_path .. flow_name)
  		--]]
  		local this_seq_gt = rgb_data.ground_truth[rgb_data.training_set[i]]
  		local seq_fea = {}
  		local img_gt_rgb = this_seq_gt[1]
  		local fn = img_gt_rgb.info.vn
  		local start_frm_id = img_gt_rgb.info.frm
  		local last_frm_id = start_frm_id + seq_len - 1
  		local video_path = data_dir .. fn .. '/'
  		--local st_1 = os.clock()
  		for j=start_frm_id, last_frm_id do
			local fea_path = video_path .. 'flow_img/' .. j .. '_flow.mat'
			--print(fea_path)
			local conv_fea = matio.load(fea_path)
			conv_fea = conv_fea.conv_fea
			conv_fea.type(torch.Tensor)
			conv_fea = conv_fea:cuda()
			table.insert(seq_fea, conv_fea)
		end
		--local et_1 = os.clock()
		--print('load time = ' .. et_1 - st_1 )
		local seq_fea_name = i .. '_flow.t7'
		local seq_fea_path = save_data_path .. seq_fea_name
		local seq_fea_tensor = func_table_2_tensor(seq_fea)
		--st_1 = os.clock()
		--print('transfer time = ' .. st_1 - et_1)
		torch.save(seq_fea_path, {seq_fea = seq_fea_tensor})
		--os.exit()
		--et_1 = os.clock()
		--print('save time = ' .. et_1 - st_1 .. ' -- ' .. et_1 - st_all)
  	end

end


function func_add_data()
	local data_root_dir = '/media/zangxh/JJH/sahasuman-bmvc2016_code-fb0de209d699/ucf_train_test_data/'
	local data_dir = data_root_dir .. 'train_test_data/'

	local save_flow_path = './data/flow_training/'
	local save_rgb_path = './data/training/'
	local seq_len = ConvGRU_rho -- gloabl variant
	local video_name_list_file = data_root_dir .. 'train_video_list.txt'
	
	local training_data = torch.load('data/training/ground_truth.t7') 
  	training_data = training_data.training_data

  	local count = #old_data.training_set+1

  	local file_count = 1
	for video_name in io.lines(video_name_list_file) do
		print('File # ' .. file_count .. ' # ' .. video_name .. ' ##\n')
		local video_path = data_dir .. video_name .. '/'
		local annot_file = video_path .. 'annot.mat'
		local annot = matio.load(annot_file)
		annot = annot.annot
		annot.type(torch.IntTensor)
		local total_frms = annot:size(1)
		local start_frm_id = 1
		while (annot[start_frm_id][6]==0) do
			start_frm_id = start_frm_id + 1
		end
		local last_frm_id = start_frm_id + seq_len -1

		while last_frm_id <= total_frms do	
			print('\t Seq ' .. count .. ' : ' ..  start_frm_id .. ' to ' .. last_frm_id .. ' :\n')
			local seq_fea = {}
			local flow_seq_fea = {}
			local seq_ground_truth = {}			
			for i=start_frm_id,last_frm_id do
				local fea_path = video_path .. i .. '.mat'
				local flow_path = video_path .. 'flow_img/' .. i .. '_flow.mat'
				--print(fea_path)
				local conv_fea = matio.load(fea_path)
				conv_fea = conv_fea.conv_fea
				conv_fea.type(torch.Tensor)
				conv_fea = conv_fea:cuda()
				table.insert(seq_fea, conv_fea)

				local flow_fea = matio.load(flow_path)
				flow_fea = flow_fea.conv_fea
				flow_fea.type(torch.Tensor)
				flow_fea = flow_fea:cuda()
				table.insert(flow_seq_fea, flow_fea)	

				local img_ground_truth = {}
				img_ground_truth.img_size = annot[i][{{2,3}}]
				img_ground_truth.rois = func_get_roi(annot[i])
				img_ground_truth.info = {vn=video_name, frm=i}

				for j=1,#img_ground_truth.rois do
					print('\t\t Class = ' .. img_ground_truth.rois[j].class_index .. ' ; Rect = [ ' .. img_ground_truth.rois[j].rect.minX .. ', ' .. img_ground_truth.rois[j].rect.minX .. ', ' .. img_ground_truth.rois[j].rect.maxX .. ', ' .. img_ground_truth.rois[j].rect.maxY .. ' ]\n')
				end

				table.insert(seq_ground_truth, img_ground_truth)
			end
			local seq_fea_name = count .. '.t7'
			local seq_fea_path = save_rgb_path .. seq_fea_name
			local seq_fea_tensor = func_table_2_tensor(seq_fea)
			torch.save(seq_fea_path, {seq_fea = seq_fea_tensor})

			local flow_fea_name = count .. '_flow.t7'
			local flow_fea_path = save_flow_path .. flow_fea_name
			local seq_flow_tensor = func_table_2_tensor(flow_seq_fea)
			torch.save(flow_fea_path, {seq_fea = seq_flow_tensor})

			table.insert(training_data.training_set, count .. '.t7')

			training_data.ground_truth[seq_fea_name] = seq_ground_truth -- file name as index

			start_frm_id = start_frm_id + torch.random(torch.Generator(),1,seq_len)
			if start_frm_id < total_frms then
				while (annot[start_frm_id][6]==0 and start_frm_id < total_frms) do
					start_frm_id = start_frm_id + 1
				end
			end
			last_frm_id = start_frm_id + seq_len - 1
			count = count + 1
		end
		file_count = file_count + 1

	end



end

-- run
func_create_training_data()

--func_create_training_flow_data()



