require 'cunn'
require 'cutorch'
require 'Hjj_BatchIterator'
require 'Hjj_Localizer'
require 'Hjj_encoder_decoder_ConvGRU'
require 'Hjj_global'

function func_get_proposals_fast(cfg, rois, prop_candidates)
  local et = os.clock()
  local neg={}
  local pos={}
  local back={}
  local candidates = prop_candidates
  if #candidates > 0 and #rois>0 then
      for k,can in ipairs(candidates) do
        local f = 0
        for l, roi in ipairs(rois) do
          if Rect.IoU(roi.rect, can.r) >= 0.5 then
             f = 1
             can.roi = roi
             break
          elseif Rect.IoU(roi.rect, can.r) > 0.1 then
             f = 2
          end
        end


        if f == 1 then
          local anchor_rect = can.r
          anchor_rect.layer = can.l
          anchor_rect.aspect = can.a
          anchor_rect.index = can.i
          table.insert(pos,{anchor_rect, can.roi})
        elseif f == 2 or k < 200 then
          local anchor_rect = can.r
          anchor_rect.layer = can.l
          anchor_rect.aspect = can.a
          anchor_rect.index = can.i
          table.insert(neg,{anchor_rect})
        else
          local anchor_rect = can.r
          anchor_rect.layer = can.l
          anchor_rect.aspect = can.a
          anchor_rect.index = can.i
          back[1] = {anchor_rect}
        end

      end

    else -- background frame
      --print('background')
      for i=1,math.min(32,#candidates) do
        local can = candidates[i]
        local anchor_rect = can.r
        anchor_rect.layer = can.l
        anchor_rect.aspect = can.a
        anchor_rect.index = can.i
        table.insert(neg,{anchor_rect})
      end
    end


  local final_neg = {}
  if #pos > 0 then
    local pos_num = #pos
    local neg_num = pos_num/cfg.fastrcnn_proportion
    if neg_num < #neg then
      for i=1,neg_num do
        table.insert(final_neg, neg[i])
      end
    elseif #neg > 0 then
      final_neg = neg
    else
      final_neg = back
    end
  else
    local neg_num = 32
    if #neg > neg_num then
      for i=1,neg_num do
         table.insert(final_neg, neg[i])
      end
    elseif #neg > 0 then
      final_neg = neg
    else
      final_neg = back
    end
  end

  local st = os.clock()
  --print('nms: ' .. st - et)
  --print('pos candidates = ' .. #pos .. ' ; neg candidates = ' .. #final_neg)
  return pos, final_neg

end

function func_get_proposals(cfg, anchors, rois, img_size, outputs, t)
  local st = os.clock()
  local neg={}
  local pos={}
  local back={}
  local prop_candidates={}
  local lsm = nn.SoftMax():cuda() -- for training it is logsoftmax
  local input_rect = Rect.new(1, 1, img_size[2], img_size[1])
  local scales = 4
  local aspect_ratios = 3

  local matches = {}
  for i=1,scales do 
    local layer = outputs[i][t]
    local layer_size = layer:size()
    for a=1,aspect_ratios do
      local ofs = (a-1) * 6
      local cls_out = layer[{{ofs + 1, ofs + 2}, {}, {}}] 
      local reg_out = layer[{{ofs + 3, ofs + 6}, {}, {}}]
      local soft_cls = lsm:forward(cls_out)
      local bool_mat = torch.gt(soft_cls[{{1},{},{}}],0.25)
      --print(bool_mat:size())
      local idx = torch.nonzero(bool_mat)
      if idx:size():size() ~= 0 then
        for j=1,idx:size(1) do
          local anc = anchors:get(i,a,idx[j][2],idx[j][3])
          local r = nil
          if cfg.finetune ~= 0 then -- train fast-rcnn
            r = Anchors.anchorToInput(anc, reg_out[{{},{idx[j][2]}, {idx[j][3]}}]:squeeze())
          else -- train the whole model
            r = Anchors.anchorToInput(anc, {0,0,0,0})
          end
          table.insert(matches, {p=soft_cls[1][idx[j][2]][idx[j][3]], r=r, l=i, a=a, i={{ofs + 1, ofs + 6},idx[j][2],idx[j][3]}})
        end
      end
    end
  end

  local et = os.clock()
  --print('get_proposal: ' .. et-st)

  if #matches > 0 then
    local bb = torch.Tensor(#matches, 4)
    local score = torch.Tensor(#matches, 1)
    for i=1,#matches do
      bb[i] = matches[i].r:totensor()
      score[i] = matches[i].p
    end
    
    local pick = nms(bb, 0.7, score)
    local nms_candidates = {}
    pick:apply(function (x) table.insert(nms_candidates, matches[x]) end )

    
    print(#nms_candidates)
    local top_n = math.min(#nms_candidates,1000)
    local candidates = {}
    if top_n == #nms_candidates then
    --if false then
      candidates = nms_candidates
    else
      for i=1,top_n do
        table.insert(candidates,nms_candidates[i])
      end
    end
    prop_candidates = candidates

    if #candidates > 0 and #rois>0 then
      for k,can in ipairs(candidates) do
        local f = 0
        for l, roi in ipairs(rois) do
          if Rect.IoU(roi.rect, can.r) >= 0.5 then
             f = 1
             can.roi = roi
             break
          elseif Rect.IoU(roi.rect, can.r) > 0.1 then
             f = 2
          end
        end


        if f == 1 then
          local anchor_rect = can.r
          anchor_rect.layer = can.l
          anchor_rect.aspect = can.a
          anchor_rect.index = can.i
          table.insert(pos,{anchor_rect, can.roi})
        elseif f == 2 or k < 200 then
          local anchor_rect = can.r
          anchor_rect.layer = can.l
          anchor_rect.aspect = can.a
          anchor_rect.index = can.i
          table.insert(neg,{anchor_rect})
        else
          local anchor_rect = can.r
          anchor_rect.layer = can.l
          anchor_rect.aspect = can.a
          anchor_rect.index = can.i
          back[1] = {anchor_rect}
        end

      end

    else -- background frame
      --print('background')
      for i=1,math.min(32,#candidates) do
        local can = candidates[i]
        local anchor_rect = can.r
        anchor_rect.layer = can.l
        anchor_rect.aspect = can.a
        anchor_rect.index = can.i
        table.insert(neg,{anchor_rect})
      end
    end

  end

  local final_neg = {}
  if #pos > 0 then
    local pos_num = #pos
    local neg_num = pos_num/cfg.fastrcnn_proportion
    if neg_num < #neg then
      for i=1,neg_num do
        table.insert(final_neg, neg[i])
      end
    elseif #neg > 0 then
      final_neg = neg
    else
      final_neg = back
    end
  else
    local neg_num = 32
    if #neg > neg_num then
      for i=1,neg_num do
         table.insert(final_neg, neg[i])
      end
    elseif #neg > 0 then
      final_neg = neg
    else
      final_neg = back
    end
  end

  local st = os.clock()
  --print('nms: ' .. st - et)
  --print('pos candidates = ' .. #pos .. ' ; neg candidates = ' .. #final_neg)
  return pos, final_neg, prop_candidates
end


function extract_roi_pooling_input(input_rect, localizer, feature_layer_output)
  --print(input_rect)
  local r = localizer:inputToFeatureRect(input_rect)
  -- the use of math.min ensures correct handling of empty rects, 
  -- +1 offset for top, left only is conversion from half-open 0-based interval
  local s = feature_layer_output:size()
  r = r:clip(Rect.new(1, 1, s[3], s[2]))
  local idx = { {}, { math.min(r.minY, r.maxY), r.maxY }, { math.min(r.minX, r.maxX), r.maxX } }
  --print(idx)
  --print(feature_layer_output:size())
  return feature_layer_output[idx], idx
end

function create_objective(model, weights, gradient, batch_iterator, stats, model_2)
  local cfg = model.cfg
  --local pnet = model.pnet
  local rpn = model.pnet.rpn
  local enc = model.pnet.enc
  local red_net = model.pnet.red_net
  local cnet = model.cnet
  local model2 = model_2
  if cfg.finetune == 1 then
    enc:evaluate()
    red_net:evaluate()
    rpn:evaluate()
    cnet:training()
  elseif cfg.finetune == 0 then
    enc:training()
    red_net:training()
    rpn:training()
    cnet:evaluate()
  elseif cfg.finetune == 3 then
    ----[[
    enc:training()
    red_net:training()
    rpn:training()
    cnet:training()
    --]]
    --[[
    enc:evaluate()
    red_net:evaluate()
    rpn:evaluate()
    cnet:training()
    --]]
    model2.pnet.rpn:evaluate()
    model2.pnet.enc:evaluate()
    model2.pnet.red_net:evaluate()
  else
    enc:training()
    red_net:training()
    rpn:training()
    cnet:training()
  end
  local bgclass = cfg.class_count + 1   -- background class
  local anchors = batch_iterator.anchors    
  local localizer = Localizer.new(rpn.outnode.children[5])
  
  local lsm = nn.LogSoftMax():cuda()  
  local softmax = nn.CrossEntropyCriterion():cuda()
  local cls_weight = torch.Tensor(cfg.class_count+1):fill(1)
  local cnll = nn.ClassNLLCriterion(cls_weight, false)
  local smoothL1_2
  if cfg.finetune == 2 then
    cutorch.withDevice(GPU_ID, function() cnll = cnll:cuda() end)
    cutorch.withDevice(GPU_ID, function() smoothL1_2 = nn.SmoothL1Criterion():cuda() end)
    smoothL1_2.sizeAverage = false
  else
    cnll = cnll:cuda()
  end
  local smoothL1 = nn.SmoothL1Criterion():cuda()
  smoothL1.sizeAverage = false
  local kh, kw = cfg.roi_pooling.kh, cfg.roi_pooling.kw
  local cnet_input_planes = model.layers[#model.layers].filters
  local amp = nn.SpatialAdaptiveMaxPooling(kw, kh):cuda()
  
  local function cleanAnchors(examples, outputs, id_in_seq)
    local i = 1
    while i <= #examples do
      local anchor = examples[i][1]
      local fmSize = outputs[anchor.layer][id_in_seq]:size()
      if anchor.index[2] > fmSize[2] or anchor.index[3] > fmSize[3] then
        table.remove(examples, i)   -- accessing would cause ouf of range exception
      else
        i = i + 1
      end 
    end
  end
  
  local function lossAndGradient(w)
      if w ~= weights then
        weights:copy(w)
      end
      gradient:zero()

      local seq_len = ConvGRU_rho --Hjj_encoder_decoder_ConvGRU.lua

      -- statistics for proposal stage      
      local cls_loss, reg_loss = 0, 0
      local cls_count, reg_count = 0, 0
      
      
      -- statistics for fine-tuning and classification stage
      local creg_loss, creg_count = 0, 0
      local ccls_loss, ccls_count = 0, 0
	  
	    local pc_true_p_n, cc_true_p, cc_true_n = 0, 0, 0 -- number of true positive and true negtives
      local class_id = cfg.class_count+1
      -- enable dropouts 
      --pnet:training()


      local t1 = os.clock()
      local batch = batch_iterator:nextTraining()
      local batch_s = os.clock()
      print('batch prepare time = ' .. batch_s-t1)

      for i,x in ipairs(batch) do
        local seq_input = x.seq_fea -- {cudaTensors}

        local red_output = red_net:forward(seq_input)
        local enc_output = enc:forward(red_output[1])

        forwardConnection(enc,rpn)
        local outputs = rpn:forward(red_output[2])

        local outputs2_1 = nil
        local outputs2 = {}
        if cfg.finetune == 3 then
          cutorch.setDevice(GPU_ID)
          local input=seq_input:clone()
          local red_output2 = model2.pnet.red_net:forward(input)
          local enc_output2 = model2.pnet.enc:forward(red_output2[1])

          forwardConnection(model2.pnet.enc,model2.pnet.rpn)
          outputs2_1 = model2.pnet.rpn:forward(red_output2[2])
          cutorch.setDevice(DEFAULT_GPU)
          for i=1,#outputs2_1 do
            table.insert(outputs2, outputs2_1[i]:clone())
          end
        end

        local delta_outputs = {}
        ----[[
        for k,out in ipairs(outputs) do
            if not delta_outputs[k] then
              delta_outputs[k] = torch.FloatTensor():cuda()
            end
            delta_outputs[k]:resizeAs(out)
            delta_outputs[k]:zero()
        end
        --]]



        local cinput = {}
        local cctarget = {}
        local crtarget = {}

        local p_table = {}
        local n_table = {}
        local roi_pool_state_table = {}
        local prop_candidates
        for j=1,seq_len do
          --print('for img ' .. j .. ' in seq\n')
          local p = x.img_set[j].positive -- get positive and negative anchors examples
          local n = x.img_set[j].negative

          if cfg.finetune ~= 0 and j == 1 then -- train fast-rcnn only
            if cfg.finetune == 3 then
              p, n, prop_candidates = func_get_proposals(cfg, anchors, x.img_set[j].rois, x.img_set[j].size, outputs2, j,p,n)
            else
              p, n, prop_candidates = func_get_proposals(cfg, anchors, x.img_set[j].rois, x.img_set[j].size, outputs, j,p,n)
            end
            
            if #x.img_set[j].rois > 0 then
              class_id = x.img_set[j].rois[1].class_index
            end
          elseif cfg.finetune ~= 0 and j > 1 then
            --print(#prop_candidates)
              p, n = func_get_proposals_fast(cfg, x.img_set[j].rois, prop_candidates)
          elseif cfg.finetune == 0 then 
            cleanAnchors(p, outputs, j)
            cleanAnchors(n, outputs, j)
          end

        
          local roi_pool_state = {}
          local input_size = x.img_set[j].img_size

          -- process positives
          if cfg.finetune ~= 0 then -- train fast-rcnn only
            for k,pp in ipairs(p) do
              local anchor = pp[1]
              local roi = pp[2]
              --print(anchor)
              local pi, idx = extract_roi_pooling_input(anchor, localizer, outputs[#outputs][j])
              --print(pi[1][1][1])
              local po = amp:forward(pi):view(kh * kw * cnet_input_planes)
              table.insert(roi_pool_state, { input = pi, input_idx = idx, anchor = anchor, reg_proposal = anchor, roi = roi, output = po:clone(), indices = amp.indices:clone() })
            end

            for k,np in ipairs(n) do
              local anchor = np[1]
              --print(anchor)
              local pi, idx = extract_roi_pooling_input(anchor, localizer, outputs[#outputs][j])
              local po = amp:forward(pi):view(kh * kw * cnet_input_planes)
              table.insert(roi_pool_state, { input = pi, input_idx = idx, output = po:clone(), indices = amp.indices:clone() })
            end

          else -- train rpn

            for k,pp in ipairs(p) do
              local anchor = pp[1]
              local roi = pp[2]
              local l = anchor.layer

              local out = outputs[l][j]
              local delta_out = delta_outputs[l][j]

              local idx = anchor.index
              local v = out[idx]
              local d = delta_out[idx]

              -- classification
              cls_loss = cls_loss + softmax:forward(v[{{1, 2}}], 1)
  			      -- count true positive
  			      -- add codes
              local tmp_softmax = lsm:forward(v[{{1, 2}}])
  			      if math.exp(tmp_softmax[1]) > 0.5 then
  				      pc_true_p_n = pc_true_p_n + 1
  			      end
              local dc = softmax:backward(v[{{1, 2}}], 1)
              d[{{1,2}}]:add(dc)

              -- box regression
              local reg_out = v[{{3, 6}}]
              local reg_target = Anchors.inputToAnchor(anchor, roi.rect):cuda()  -- regression target
              local reg_proposal = Anchors.anchorToInput(anchor, reg_out)
              reg_loss = reg_loss + smoothL1:forward(reg_out, reg_target) * 10
              local dr = smoothL1:backward(reg_out, reg_target) * 10
              d[{{3,6}}]:add(dr)

              -- pass through adaptive max pooling operation
              local pi, idx = extract_roi_pooling_input(reg_proposal, localizer, outputs[#outputs][j])
              local po = amp:forward(pi):view(kh * kw * cnet_input_planes)
              table.insert(roi_pool_state, { input = pi, input_idx = idx, anchor = anchor, reg_proposal = reg_proposal, roi = roi, output = po:clone(), indices = amp.indices:clone() })

            end -- for k,pp in ipairs(p)

            -- process negative
            for k,np in ipairs(n) do
              local anchor = np[1]
              local l = anchor.layer
              local out = outputs[l][j]
              local delta_out = delta_outputs[l][j]
              local idx = anchor.index
              local v = out[idx]
              local d = delta_out[idx]
              
              cls_loss = cls_loss + softmax:forward(v[{{1, 2}}], 2)
              local dc = softmax:backward(v[{{1, 2}}], 2)
  			      -- add codes
              local tmp_softmax = lsm:forward(v[{{1, 2}}])
              if math.exp(tmp_softmax[2]) > 0.5 then
                pc_true_p_n = pc_true_p_n + 1
              end
              d[{{1,2}}]:add(dc)
              
              -- pass through adaptive max pooling operation
              local pi, idx = extract_roi_pooling_input(anchor, localizer, outputs[#outputs][j])
              --print(anchor)
              local po = amp:forward(pi):view(kh * kw * cnet_input_planes)
              table.insert(roi_pool_state, { input = pi, input_idx = idx, output = po:clone(), indices = amp.indices:clone() })
            end -- for k,np in ipairs(n)

          end -- if finetune == 1


          -- fine-tuning STAGE
          -- pass extracted roi-data through classification network
          -- create cnet input batch
          if cfg.finetune == 2 then
            if #roi_pool_state > 0 then         
              cinput[j] = cutorch.withDevice(GPU_ID, function() return torch.CudaTensor(#roi_pool_state, kh * kw * cnet_input_planes) end)
              cctarget[j] = cutorch.withDevice(GPU_ID, function() return torch.CudaTensor(#roi_pool_state) end)
              --crtarget[j] = cutorch.withDevice(GPU_ID, function() return torch.CudaTensor(#roi_pool_state, 4):zero() end)
              crtarget[j] = cutorch.withDevice(GPU_ID, function() return torch.CudaTensor(#roi_pool_state, 4*bgclass):zero() end)
            
              for k,rps in ipairs(roi_pool_state) do
                cinput[j][k] = rps.output
                if rps.roi then
                  -- positive example
                  cctarget[j][k] = rps.roi.class_index
                  crtarget[j][{{k}, {(rps.roi.class_index-1)*4+1,rps.roi.class_index*4}}] = Anchors.inputToAnchor(rps.reg_proposal, rps.roi.rect)   -- base fine tuning on proposal
                  --crtarget[j][k] = Anchors.inputToAnchor(rps.reg_proposal, rps.roi.rect)   -- base fine tuning on proposal
                else
                  -- negative example
                  cctarget[j][k] = bgclass
                end
              end

            else
              cinput[j] = cutorch.withDevice(GPU_ID, function() return torch.CudaTensor(1, kh * kw * cnet_input_planes):fill(0) end)
              cctarget[j] = cutorch.withDevice(GPU_ID, function() return torch.CudaTensor(1):fill(bgclass) end)
              crtarget[j] = cutorch.withDevice(GPU_ID, function() return torch.CudaTensor(1, 4*bgclass):zero() end)
              --crtarget[j] = cutorch.withDevice(GPU_ID, function() return torch.CudaTensor(1, 4):zero() end)
            end
          else
            if #roi_pool_state > 0 then    
              cinput[j] = torch.CudaTensor(#roi_pool_state, kh * kw * cnet_input_planes)
              cctarget[j] = torch.CudaTensor(#roi_pool_state)
              crtarget[j] = torch.CudaTensor(#roi_pool_state, 4*bgclass):zero()
              --crtarget[j] = torch.CudaTensor(#roi_pool_state, 4):zero()
              for k,rps in ipairs(roi_pool_state) do
                cinput[j][k] = rps.output
                if rps.roi then
                  -- positive example
                  cctarget[j][k] = rps.roi.class_index
                  crtarget[j][{ {k}, {(rps.roi.class_index-1)*4+1,rps.roi.class_index*4}}] = Anchors.inputToAnchor(rps.reg_proposal, rps.roi.rect)   -- base fine tuning on proposal
                else
                  -- negative example
                  cctarget[j][k] = bgclass
                end
              end

            else
              cinput[j] = torch.CudaTensor(1, kh * kw * cnet_input_planes):fill(0)
              cctarget[j] = torch.CudaTensor(1):fill(bgclass)
              crtarget[j] = torch.CudaTensor(1, 4*bgclass):zero()
              print('shichuile')
              --crtarget[j] = torch.CudaTensor(1, 4):zero()
            end            
          end -- if #roi_pool_state > 0 then

          table.insert(p_table, p)
          table.insert(n_table, n)
          table.insert(roi_pool_state_table, roi_pool_state)
          reg_count = reg_count + #p
          cls_count = cls_count + #p + #n
        
          creg_count = creg_count + #p
          ccls_count = ccls_count + #p + #n
        end -- for j=1,seq_len do 

        -- process classification batch 
        if cfg.finetune ~= 0 then
          local coutputs
          if cfg.finetune == 2 then
            cutorch.withDevice(GPU_ID, function() coutputs=cnet:forward(cinput) end)
          else
            coutputs=cnet:forward(cinput)
          end

          local crdelta = {}
          local ccdelta = {}

          for j=1,seq_len do 
            local crout = coutputs[1][j]
            local ccout = coutputs[2][j]  -- log softmax classification
            --print(ccout)

            --print('p -- n '.. #p_table[j] .. ' - ' .. #n_table[j])
            if #roi_pool_state_table[j] == 0 then
              if cfg.finetune == 2 then
                cutorch.withDevice(GPU_ID, function() crout[{{1, #roi_pool_state_table[j]+1}, {}}]:zero() end) -- ignore 
                cutorch.withDevice(GPU_ID, function() ccout[{{1, #roi_pool_state_table[j]+1}, {}}]:zero() end)-- ignore
              else
                crout[{{1, #roi_pool_state_table[j]+1}, {}}]:zero() -- ignore 
                ccout[{{1, #roi_pool_state_table[j]+1}, {}}]:zero() -- ignore
              end
            elseif #roi_pool_state_table[j] > #p_table[j] then
              if cfg.finetune == 2 then
                cutorch.withDevice(GPU_ID, function() crout[{{#p_table[j] + 1, #roi_pool_state_table[j]}, {}}]:zero() end) -- ignore negative examples
              else
                crout[{{#p_table[j] + 1, #roi_pool_state_table[j]}, {}}]:zero()
              end
            end 
            -- compute classification and regression error and run backward pass 
            if cfg.finetune == 2 then
              cutorch.withDevice(GPU_ID, function()  creg_loss = creg_loss + smoothL1_2:forward(crout, crtarget[j]) * 10 end)
              cutorch.withDevice(GPU_ID, function() table.insert(crdelta, smoothL1_2:backward(crout, crtarget[j]) * 10) end)
            else
              creg_loss = creg_loss + smoothL1:forward(crout, crtarget[j]) * 10
              table.insert(crdelta, smoothL1:backward(crout, crtarget[j]) * 10)
            end
            
  		      -- add codes

            local loss
            if cfg.finetune == 2 then
              cutorch.withDevice(GPU_ID, function() loss = cnll:forward(ccout, cctarget[j]) end)
              ccls_loss = ccls_loss + loss 
              local tmp_ccdelta = cutorch.withDevice(GPU_ID, function() return cnll:backward(ccout, cctarget[j]) end)
              table.insert(ccdelta, cutorch.withDevice(GPU_ID, function() return tmp_ccdelta:clone() end))
              local cprob = cutorch.withDevice(GPU_ID, function() return ccout end)
              local p,c = cutorch.withDevice(GPU_ID, function() return torch.max(cprob, 2) end ) -- get probabilities and class indicies
              
              for lala = 1, p:size(1) do
                if c[lala][1] ==  cctarget[j][lala] then
                  if c[lala][1] == bgclass then
                    cc_true_n = cc_true_n + 1
                  else
                    cc_true_p = cc_true_p + 1
                  end
                end
              end
            else
              loss = cnll:forward(ccout, cctarget[j])
              ccls_loss = ccls_loss + loss 
              local tmp_ccdelta = cnll:backward(ccout, cctarget[j])
              table.insert(ccdelta, tmp_ccdelta:clone())
              local cprob = ccout
              local p,c = torch.max(cprob, 2) -- get probabilities and class indicies
              
              for lala = 1, p:size(1) do
                --print('(' .. c[lala][1] .. ',' .. cctarget[j][lala] .. ')')
                if c[lala][1] ==  cctarget[j][lala] then
                  if c[lala][1] == bgclass then
                    cc_true_n = cc_true_n + 1
                    Bg_Table[class_id] = Bg_Table[class_id] +1
                    Neg_Prop_Table[class_id] = Neg_Prop_Table[class_id] + 1
                  else
                    cc_true_p = cc_true_p + 1
                    Det_Table[class_id]  = Det_Table[class_id] +1
                    Pos_Prop_Table[class_id] = Pos_Prop_Table[class_id] + 1
                  end
                
                else
                  if cctarget[j][lala] == bgclass then
                    Neg_Prop_Table[class_id] = Neg_Prop_Table[class_id] + 1
                  else
                    Pos_Prop_Table[class_id] = Pos_Prop_Table[class_id] + 1
                  end
                end
              end

            end

          end --for j=1,seq_len do 

          local post_roi_delta
          if cfg.finetune == 2 then
              cutorch.withDevice(GPU_ID, function() post_roi_delta = cnet:backward(cinput, { crdelta, ccdelta }) end)

          else
            post_roi_delta = cnet:backward(cinput, { crdelta, ccdelta })
          end
          -- pass the multi-class net gradients backward to rpn
        
          for j=1,seq_len do
            if #roi_pool_state_table[j] > 0 then
              for k, rps in ipairs(roi_pool_state_table[j]) do
                amp.indices = rps.indices
                delta_outputs[5][j][rps.input_idx]:add(amp:backward(rps.input, post_roi_delta[j][k]:view(cnet_input_planes, kh, kw):clone() ))
              end
            end
          end

        end -- process classification batch end

        -- run rpn backward propagation
        if cfg.finetune ~= 1 then
          -- backward pass of proposal network
          local delta_rpn = rpn:backward(red_output[2], delta_outputs)
          backwardConnection(enc, rpn)
          local delta_enc_input = torch.FloatTensor():cuda()
          delta_enc_input:resizeAs(enc_output)
          delta_enc_input:fill (0)
          local delta_enc = enc:backward(red_output[1],delta_enc_input)

          -- backward pass of red_net
          local delta_red_net = red_net:backward(seq_input, {delta_enc, delta_rpn})
        end


      end -- for i,x in ipairs(batch) do

      -- scale gradient
      if cls_count == 0 then
        cls_count = 1
        gradient:zero()
      end
      gradient:div(cls_count)
      if cfg.finetune ~= 1 then
        gradient:clamp(-50,50)
      end

      local pcls = cls_loss / cls_count     -- proposal classification (bg/fg)
      local preg = reg_loss / reg_count     -- proposal bb regression
      local dcls = ccls_loss / ccls_count   -- detection classification
      local dreg = creg_loss / creg_count   -- detection bb finetuning

      
      print(string.format('prop: cls: %f (%d), reg: %f (%d); det: cls: %f, reg: %f', 
        pcls, cls_count, preg, reg_count, dcls, dreg)
      )
      print('pos vs all = ' .. reg_count/cls_count)
      print('precision: pc = ' .. pc_true_p_n/cls_count .. ' ; class = ' .. class_id ..' ; cc = ' .. cc_true_p/reg_count .. ' - ' .. cc_true_n/(cls_count - reg_count))
      
      table.insert(stats.pcls, pcls)
      table.insert(stats.preg, preg)
      table.insert(stats.dcls, dcls)
      table.insert(stats.dreg, dreg)

      local batch_e = os.clock()
      print('batch time: ' .. batch_e - batch_s)
      
      local loss = pcls + preg + dcls + dreg
      return loss, gradient
  end
    
  return lossAndGradient
end

