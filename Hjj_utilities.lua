require 'lfs' -- lua file system for directory listings
require 'nn'
require 'cunn'
require 'image'
require 'Hjj_global'

function list_files(directory_path, max_count, abspath)
  local l = {}
  for fn in lfs.dir(directory_path) do
    if max_count and #l >= max_count then
      break
    end
    local full_fn = path.join(directory_path, fn)
    if lfs.attributes(full_fn, 'mode') == 'file' then 
      table.insert(l, abspath and full_fn or fn)
    end
  end
  return l
end

function clamp(x, lo, hi)
  return math.max(math.min(x, hi), lo)
end

function saturate(x)
  return clam(x, 0, 1)
end

function lerp(a, b, t)
  return (1-t) * a + t * b
end

function shuffle_n(array, count)
  count = math.max(count, count or #array)
  local r = #array    -- remaining elements to pick from
  local j, t
  for i=1,count do
    j = math.random(r) + i - 1
    t = array[i]    -- swap elements at i and j
    array[i] = array[j]
    array[j] = t
    r = r - 1
  end
end

function shuffle(array)
  local i, t
  for n=#array,2,-1 do
    i = math.random(n)
    t = array[n]
    array[n] = array[i]
    array[i] = t
  end
  return array
end

function shallow_copy(t)
  local t2 = {}
  for k,v in pairs(t) do
    t2[k] = v
  end
  return t2
end

function deep_copy(obj, seen)
  if type(obj) ~= 'table' then 
    return obj 
  end
  if seen and seen[obj] then 
    return seen[obj] 
  end
  local s = seen or {}
  local res = setmetatable({}, getmetatable(obj))
  s[obj] = res
  for k, v in pairs(obj) do 
    res[deep_copy(k, s)] = deep_copy(v, s) 
  end
  return res
end

function reverse(array)
  local n = #array, t 
  for i=1,n/2 do
    t = array[i]
    array[i] = array[n-i+1]
    array[n-i+1] = t
  end
  return array
end

function remove_tail(array, num)
  local t = {}
  for i=num,1,-1 do
    t[i] = table.remove(array)
  end
  return t, array
end

function keys(t)
  local l = {}
  for k,v in pairs(t) do
    table.insert(l, k)
  end
  return l
end

function values(t)
  local l = {}
  for k,v in pairs(t) do
    table.insert(l, v)
  end
  return l
end

function save_obj(file_name, obj)
  local f = torch.DiskFile(file_name, 'w')
  f:writeObject(obj)
  f:close()
end

function load_obj(file_name)
  local f = torch.DiskFile(file_name, 'r')
  local obj = f:readObject()
  f:close()
  return obj
end

function save_model(file_name, weights, options, stats)
  save_obj(file_name,
  {
    version = 0,
    weights = weights,
    options = options,
    stats = stats
  })
end

function combine_and_flatten_parameters(...)
  local nets = { ... }
  local parameters,gradParameters = {}, {}
  for i=1,#nets do
    local w, g = nets[i]:parameters()
    for i=1,#w do
      table.insert(parameters, w[i])
      table.insert(gradParameters, g[i])
    end
  end
  return nn.Module.flatten(parameters), nn.Module.flatten(gradParameters)
end

function draw_rectangle(img, rect, color)
  local sz = img:size()
  
  local x0 = math.max(1, rect.minX)
  local x1 = math.min(sz[3], rect.maxX)
  local w = math.floor(x1) - math.floor(x0)
  if w >= 0 then
    local v = color:view(3,1):expand(3, w + 1)
    if rect.minY > 0 and rect.minY <= sz[2] then
      img[{{}, rect.minY, {x0, x1}}] = v
    end
    if rect.maxY > 0 and rect.maxY <= sz[2] then
      img[{{}, rect.maxY, {x0, x1}}] = v
    end
  end
  
  local y0 = math.max(1, rect.minY)
  local y1 = math.min(sz[2], rect.maxY)
  local h = math.floor(y1) - math.floor(y0)
  if h >= 0 then
    local v = color:view(3,1):expand(3, h + 1)
    if rect.minX > 0 and rect.minX <= sz[3] then
      img[{{}, {y0, y1}, rect.minX}] = v 
    end
    if rect.maxX > 0 and rect.maxX <= sz[3] then
      img[{{}, {y0, y1}, rect.maxX}] = v
    end
  end
end

function remove_quotes(s)
  return s:gsub('^"(.*)"$', "%1")
end

function normalize_debug(t)
  local lb, ub = t:min(), t:max()
  return (t -lb):div(ub-lb+1e-10)
end

function find_target_size(orig_w, orig_h, target_smaller_side, max_pixel_size)
  local w, h
  if orig_h < orig_w then
    -- height is smaller than width, set h to target_size
    w = math.min(orig_w * target_smaller_side/orig_h, max_pixel_size)
    h = math.floor(orig_h * w/orig_w + 0.5)
    w = math.floor(w + 0.5)
  else
    -- width is smaller than height, set w to target_size
    h = math.min(orig_h * target_smaller_side/orig_w, max_pixel_size)
    w = math.floor(orig_w * h/orig_h + 0.5)
    h = math.floor(h + 0.5)
  end
  assert(w >= 1 and h >= 1)
  return w, h
end

function load_image(fn, color_space, base_path)
  if not path.isabs(fn) and base_path then
    fn = path.join(base_path, fn)
  end
  local img = image.load(fn, 3, 'float')
  if color_space == 'yuv' then
    img = image.rgb2yuv(img)
  elseif color_space == 'lab' then
    img = image.rgb2lab(img)
  elseif color_space == 'hsv' then
    img = image.rgb2hsv(img)
  end
  return img
end

function func_get_seq_fea(fn, base_path)
  if not path.isabs(fn) and base_path then
    fn = path.join(base_path, fn)
  end
  local seq_fea = torch.load(fn)
  seq_fea = seq_fea.seq_fea -- CudaTensors in table
  return seq_fea[{{1,ConvGRU_rho},{},{},{}}]
end

function func_table_2_tensor(seq_fea_table)
	local seq_fea_tensor = torch.Tensor(#seq_fea_table,seq_fea_table[1]:size(1),seq_fea_table[1]:size(2),seq_fea_table[1]:size(3)):cuda()
	for i=1,#seq_fea_table do
		seq_fea_tensor[i] = seq_fea_table[i]
	end
	return seq_fea_tensor
end

function func_processRoIs(cfg,img_size) -- transform the rois rect according to the transformation of faster-rcnn input
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

  return img_size, resize_scale
end

function func_collect_frm_result(pre_video_result, frm_result, start_frm_id)
  local video_result_table = {}
  local video_result
  for i,v in ipairs(frm_result) do
      for j, x in ipairs(v) do
        table.insert(video_result_table, {start_frm_id+i-1, x.class, math.exp(x.confidence),x.p, x.r.minY, x.r.minX, x.r.maxY, x.r.maxX, x.r2.minY, x.r2.minX, x.r2.maxY, x.r2.maxX})
      end
  end

  if pre_video_result then
    if #video_result_table > 0 then
      video_result = torch.Tensor(#video_result_table+pre_video_result:size(1),#video_result_table[1])
      video_result[{{1,pre_video_result:size(1)},{}}] = pre_video_result
      for i=1, #video_result_table do
          for j=1, #video_result_table[1] do
          video_result[i+pre_video_result:size(1)][j] = video_result_table[i][j]
        end
      end 
    else
      video_result = pre_video_result      
    end

  else
    -- pre_video_result is nil
    if #video_result_table > 0 then
      video_result = torch.Tensor(#video_result_table,#video_result_table[1]):fill(0)
      for i=1, #video_result_table do
          for j=1, #video_result_table[1] do
            video_result[i][j] = video_result_table[i][j]
          end
      end
    end
  end
  return video_result
end

function func_hmdb_collect_frm_result(pre_video_result, frm_result, start_frm_id,resize_scale)
  local video_result_table = {}
  local video_result
  for i,v in ipairs(frm_result) do
      for j, x in ipairs(v) do
        table.insert(video_result_table, {start_frm_id+i-1, x.class, math.exp(x.confidence),x.p, x.r.minY/resize_scale, x.r.minX/resize_scale, x.r.maxY/resize_scale, x.r.maxX/resize_scale, x.r2.minY/resize_scale, x.r2.minX/resize_scale, x.r2.maxY/resize_scale, x.r2.maxX/resize_scale})
      end
  end

  if pre_video_result then
    if #video_result_table > 0 then
      video_result = torch.Tensor(#video_result_table+pre_video_result:size(1),#video_result_table[1])
      video_result[{{1,pre_video_result:size(1)},{}}] = pre_video_result
      for i=1, #video_result_table do
          for j=1, #video_result_table[1] do
          video_result[i+pre_video_result:size(1)][j] = video_result_table[i][j]
        end
      end 
    else
      video_result = pre_video_result      
    end

  else
    -- pre_video_result is nil
    if #video_result_table > 0 then
      video_result = torch.Tensor(#video_result_table,#video_result_table[1]):fill(0)
      for i=1, #video_result_table do
          for j=1, #video_result_table[1] do
            video_result[i][j] = video_result_table[i][j]
          end
      end
    end
  end
  return video_result
end

function func_get_center(map, scale)
  local max,miny,maxy,minx,maxx = 0, 0, 0, 0, 0
  local a,b = torch.max(map,2)
  max, miny = torch.max(a,1)

  max=max[1][1]
  miny=miny[1][1]

  for i=a:size(1),1,-1 do
    if a[i][1] >= max then
        maxy = i
        break
    end
  end

  a,b = torch.max(map,1)
  max, minx = torch.max(a,2)

  max=max[1][1]
  minx=minx[1][1]
  for i=a:size(2),1,-1 do
    if a[1][i] >= max then
      maxx = i
      break
    end
  end
  
  return minx, miny, maxx, maxy
end

function func_get_nearby_center(x,y,fea_size)
  local x_y_pairs = {}
  local max_stride = 4 -- max_stride*16 in image
  --table.insert(x_y_pairs, {x=x,y=y})
  local minx =math.min(math.max(1,x-max_stride),fea_size[3]) 
  local miny =math.min(math.max(1,y-max_stride),fea_size[2]) 
  local maxx = math.min(fea_size[3],math.max(1,x+max_stride))
  local maxy = math.min(fea_size[2],math.max(1,y+max_stride))
  for i=minx, maxx do
    for j = miny, maxy do
      table.insert(x_y_pairs,{x=i,y=j})
    end
  end
  return x_y_pairs
end

function func_fea_overlay(rgb_fea, flow_fea)
  if rgb_fea:size():size() == 4 then
    local fea = torch.Tensor(rgb_fea:size(1),rgb_fea:size(2)*2,rgb_fea:size(3), rgb_fea:size(4)):cuda()
    fea[{{},{1,rgb_fea:size(2)},{},{}}] = rgb_fea
    fea[{{},{rgb_fea:size(2)+1,rgb_fea:size(2)*2},{},{}}] = flow_fea
    return fea
  else
    local fea = torch.Tensor(rgb_fea:size(1)*2,rgb_fea:size(2), rgb_fea:size(3)):cuda()
    fea[{{1,rgb_fea:size(1)},{},{}}] = rgb_fea
    fea[{{rgb_fea:size(1)+1,rgb_fea:size(1)*2},{},{}}] = flow_fea
    return fea
  end
end


















