require 'Rect'
require 'cutorch'
require 'math'

function func_score_of_edge(node1, node2)
  local edge_score = torch.Tensor(#node1.sub_nodes, #node2.sub_nodes):fill(0)
  for i=1, #node1.sub_nodes do
    for j=1, #node2.sub_nodes do 
    	local iou = Rect.IoU(node1.sub_nodes[i].r, node2.sub_nodes[j].r)
    	--edge_score[i][j] = Rect.IoU(node1.sub_nodes[i].r, node2.sub_nodes[j].r) + Rect.Shape_similarity(node1.sub_nodes[i].r, node2.sub_nodes[j].r) + node1.sub_nodes[i].p + node2.sub_nodes[j].p
    	edge_score[i][j] =  0.5 * iou + (node1.sub_nodes[i].p + node2.sub_nodes[j].p)*10 + torch.sigmoid(-torch.dist(node1.sub_nodes[i].roi_fea, node2.sub_nodes[j].roi_fea)/torch.norm(node1.sub_nodes[i].roi_fea)+1)*4

    	--if iou > 0.3 then
    	--	edge_score[i][j] =  (node1.sub_nodes[i].p + node2.sub_nodes[j].p)
    	--end
    	--print(Rect.IoU(node1.sub_nodes[i].r, node2.sub_nodes[j].r) .. '--' .. node1.sub_nodes[i].p .. '--' .. node2.sub_nodes[j].p)
    end
  end
  return edge_score 
end

function func_path_planning(chain, path_available)
  local path = {}


  local chain_num = #chain[1]
  local chain_len = #chain

  --print('chain_num: ' .. chain_num)
  --print('chain_len: ' .. chain_len)
  --print(chain[1])

  for i=1, chain_num do
    -- init record table for dynamic programming
    local data = {}

    for j=1, chain_len do
    	--print(i .. ' - ' .. j)
      if chain[j][i] then
      	local can_num = #chain[j][i].sub_nodes 
      	--print(can_num)
      	table.insert(data,{score = torch.Tensor(1,can_num):fill(0), index = torch.Tensor(1,can_num):fill(0)})
      else
      	  break
      end
    end

    --print('#data : ' .. #data)

    --if #data == 1 and path_available[i] == 1 then
    if #data == 1 then
      chain[1][i].sub_nodes[1].class2 = chain[1][i].sub_nodes[1].class
      chain[1][i].sub_nodes[1].confidence2 = chain[1][i].sub_nodes[1].confidence
      table.insert(path,{nodes = chain[1][i].sub_nodes, score = chain[1][i].sub_nodes[1].p})
      
    --elseif path_available[i] == 1 or #data == chain_len then
    elseif #data == chain_len then
	    for j=#data-1, 1, -1 do
	      local edge_score = func_score_of_edge(chain[j][i], chain[j+1][i])
	      edge_score:add(torch.expand(data[j+1].score,edge_score:size(1),edge_score:size(2)))
	      local score,idx = torch.max(edge_score,2)
	      data[j].score = score:t():clone()
	      data[j].index = idx:t():clone()
	    end

	    -- viterbi
	    local chain_nodes = {}
	    local total_score = 0
	    local score, idx = data[1].score:sort()
	    idx = idx[1][1]
	    total_score = score[1][1]
	    local class_voting_score = chain[1][i].sub_nodes[idx].softmax:clone()
	    table.insert(chain_nodes, chain[1][i].sub_nodes[idx])

	    for j=1,#data-1 do
	    	idx = data[j].index[1][idx]
	    	class_voting_score = class_voting_score + chain[1+j][i].sub_nodes[idx].softmax
	    	table.insert(chain_nodes, chain[1+j][i].sub_nodes[idx])
	    end

	    class_voting_score = class_voting_score/(#data)
	    local p,c = torch.sort(class_voting_score, 1, true)
	    local class2 = c[1]
	    local confidence2 = p[1]

	    for j=1,#chain_nodes do
	    	chain_nodes[j].class2 = class2
	    	chain_nodes[j].confidence2 = confidence2
	    end

	    table.insert(path,{nodes=chain_nodes,score=total_score/(#data-1)})
	end -- if #data == 1
  end --for i=1,chain_num

  return path
end




function func_tube_window_mapping(path_buff, segs, top_n, start_frm_id)
	local updated_buff = {}
	local to_be_update_buff = {}
	local score_record = {}
	if #segs == 0 then
		return path_buff
	end

	if #path_buff == 0 then
		for i,seg in ipairs(segs) do
			table.insert(updated_buff,seg)
		end
		return updated_buff
	end
	local max_len = 1
	for i,path in ipairs(path_buff) do
		if path.nodes[#path.nodes].frm ~= start_frm_id then
			table.insert(updated_buff, path)
			table.insert(score_record, path.score)
		else
			table.insert(to_be_update_buff, path)
			if #path.nodes > max_len then
				max_len=#path.nodes
			end
		end
		
	end
	if #to_be_update_buff>0 then
		local score_table = torch.Tensor(#to_be_update_buff, #segs+1):fill(0)
		--print(score_table:size())
		score_table[{{},{#segs+1}}]:fill(1) -- zero mapping column
		local score_segs = torch.Tensor(#segs):fill(0)
		local seg_max_iou = torch.Tensor(#segs):fill(0)

		for i, path in ipairs(to_be_update_buff) do
			local node1 = path.nodes[#path.nodes]
			for j, seg in ipairs(segs) do
				local node2 = seg.nodes[1]
				local iou = Rect.IoU(node1.r, node2.r)
				if iou > seg_max_iou[j] then
					seg_max_iou[j] = iou
				end
				--print('#iou: ' .. iou)
				if iou > 0.2 then
					--score_table[i][j] = iou + Rect.Shape_similarity(node1.r, node2.r) + path.score + seg.score
					score_table[i][j] = 1 * iou + (path.score*(#path.nodes/max_len) + seg.score)*5 + torch.sigmoid(-torch.dist(node1.roi_fea, node2.roi_fea)/torch.norm(node1.roi_fea)+1)*0
					--score_table[i][j] = path.score + seg.score


				end
			end
		end

		--print(score_table)

		for i = 1,score_table:size(1) do
			local score_v,idx_v = torch.max(score_table,2)
		    local score, idx = torch.max(score_v,1)
		    local path_idx = idx[1][1]
		    local seg_idx = idx_v[path_idx][1]
		    --print('#seg_idx -- #seg: ' .. seg_idx .. ' -- ' .. #segs)
		    if seg_idx <= #segs then
		    	local path_len = #to_be_update_buff[path_idx].nodes
			    local seg_len = #segs[seg_idx].nodes
			    local path_score = to_be_update_buff[path_idx].score
			    local seg_score = segs[seg_idx].score
		    	-- mapping to nonzero column
			    if to_be_update_buff[path_idx].nodes[path_len].p < segs[seg_idx].nodes[1].p then
			    	table.remove(to_be_update_buff[path_idx].nodes,path_len)
			    	for j=1, seg_len do
			    		table.insert(to_be_update_buff[path_idx].nodes,segs[seg_idx].nodes[j])
			    	end

			    	to_be_update_buff[path_idx].score = (path_score * (path_len-1) + seg_score * seg_len)/(path_len+seg_len-1)
			    else
			    	for j=2, seg_len do
			    		table.insert(to_be_update_buff[path_idx].nodes,segs[seg_idx].nodes[j])
			    	end
			    	to_be_update_buff[path_idx].score = (path_score * path_len + seg_score * (seg_len-1))/(path_len+seg_len-1)
			    end
		     	score_table[{{},{seg_idx}}]:fill(-1)
		     	score_segs[seg_idx] = -1
			end
			score_table[path_idx]:fill(-1)
		    table.insert(updated_buff, to_be_update_buff[path_idx])
		    table.insert(score_record, to_be_update_buff[path_idx].score) 
		end

		-- dealing with unmapped window
		for i=1,score_segs:size(1) do
			if score_segs[i] > -1 and seg_max_iou[i] < 0.1 then
				-- add new tubes
				table.insert(updated_buff, segs[i])
		    	table.insert(score_record, segs[i].score)
			end
		end
	
	else
		-- old chain ended
		for i,seg in ipairs(segs) do
			table.insert(updated_buff,seg)
		end
	end -- if #t_be_update > 0
	-- check if exceed buff size

	--[[
	print(#updated_buff .. ' -- ' .. 30)
	if #updated_buff > top_n then
		local score_record_tensor = torch.Tensor(score_record)
		local s,idx = torch.sort(score_record_tensor)
		local new_table = {}
		for i=#updated_buff-top_n+1,#updated_buff do
			table.insert(new_table, updated_buff[idx[i] ])
		end
		--print('#buff3: ' .. #new_table)
		return new_table
	else
		return updated_buff
	end
	--]]
	return updated_buff, score_record
end


---------------------------------------------------------------------------------------------------------------------







function func_score_of_edge2(node1, node2)
  local edge_score = torch.Tensor(#node1, #node2):fill(0)
  for i=1, #node1 do
    for j=1, #node2 do 
    	local iou = Rect.IoU(node1[i].r, node2[j].r)
    	--edge_score[i][j] = Rect.IoU(node1.sub_nodes[i].r, node2.sub_nodes[j].r) + Rect.Shape_similarity(node1.sub_nodes[i].r, node2.sub_nodes[j].r) + node1.sub_nodes[i].p + node2.sub_nodes[j].p
    	edge_score[i][j] =  1 * iou + (node1[i].p + node2[j].p)*1 + torch.sigmoid(-torch.dist(node1[i].roi_fea, node2[j].roi_fea)/torch.norm(node1[i].roi_fea)+1)*1

    	--if iou > 0.3 then
    	--	edge_score[i][j] =  (node1.sub_nodes[i].p + node2.sub_nodes[j].p)
    	--end
    	--print(Rect.IoU(node1.sub_nodes[i].r, node2.sub_nodes[j].r) .. '--' .. node1.sub_nodes[i].p .. '--' .. node2.sub_nodes[j].p)
    end
  end
  return edge_score 
end


function func_path_all_planning(chain)
  local path = {}
  local tmp_chain = chain

  local chain_num = #chain[1]
  local chain_len = #chain

  for i=1, chain_num do
    -- init record table for dynamic programming
    local data = {}

    for j=1, chain_len do
    	--print(i .. ' - ' .. j)
      if #tmp_chain[j] > 0 then
      	local can_num = #tmp_chain[j] 
      	--print(can_num)
      	table.insert(data,{score = torch.Tensor(1,can_num):fill(0), index = torch.Tensor(1,can_num):fill(0)})
      else
      	--print('ended viterbi')
      	  return path
      end
    end

    --print('#data : ' .. #data)

    if #data ~= chain_len then
      print('error')
    else
	    for j=#data-1, 1, -1 do
	      local edge_score = func_score_of_edge2(tmp_chain[j], tmp_chain[j+1])
	      edge_score:add(torch.expand(data[j+1].score,edge_score:size(1),edge_score:size(2)))
	      local score,idx = torch.max(edge_score,2)
	      data[j].score = score:t():clone()
	      data[j].index = idx:t():clone()
	    end

	    -- viterbi
	    local chain_nodes = {}
	    local total_score = 0
	    local score, idx = data[1].score:sort()
	    idx = idx[1][1]
	    total_score = score[1][1]
	    local class_voting_score = tmp_chain[1][idx].softmax:clone()
	    table.insert(chain_nodes, tmp_chain[1][idx])
	    table.remove(tmp_chain[1],idx)

	    for j=1,#data-1 do
	    	idx = data[j].index[1][idx]
	    	class_voting_score = class_voting_score + tmp_chain[1+j][idx].softmax
	    	table.insert(chain_nodes, tmp_chain[1+j][idx])
	    	table.remove(tmp_chain[1+j],idx)
	    end

	    class_voting_score = class_voting_score/(#data)
	    local p,c = torch.sort(class_voting_score, 1, true)
	    local class2 = c[1]
	    local confidence2 = p[1]

	    for j=1,#chain_nodes do
	    	chain_nodes[j].class2 = class2
	    	chain_nodes[j].confidence2 = confidence2
	    end

	    table.insert(path,{nodes=chain_nodes,score=total_score/(#data-1)})
	end -- if #data == 1
  end --for i=1,chain_num

  return path
end




