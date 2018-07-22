require 'nngraph'
require 'cunn'
require 'cutorch'
require 'Hjj_encoder_decoder_ConvGRU'

function create_proposal_net(layers, anchor_nets, cfg)
  -- define  building block functions first

  -- VGG style 3x3 convolution building block
  local function ConvPReLU(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout)
    container:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kW,kH, 1,1, padW,padH))
    container:add(nn.PReLU())
    if dropout and dropout > 0 then
      container:add(nn.SpatialDropout(dropout))
    end
    return container
  end
  
  -- multiple convolution layers followed by a max-pooling layer
  local function ConvPoolBlock(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout, conv_steps, enable_pooling)
    for i=1,conv_steps do
      ConvPReLU(container, nInputPlane, nOutputPlane, kW, kH, padW, padH, dropout)
      nInputPlane = nOutputPlane
      dropout = nil -- only one dropout layer per conv-pool block 
    end
    if enable_pooling ~= 1 then
      container:add(nn.SpatialMaxPooling(2, 2, 2, 2):ceil())
    end
    return container
  end  
  
  -- creates an anchor network which reduces the input first to 256 dimensions 
  -- and then further to the anchor outputs for 3 aspect ratios 
  local function AnchorNetwork(nInputPlane, n, kernelWidth)
    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(nInputPlane, n, kernelWidth,kernelWidth, 1,1))
    net:add(nn.PReLU())
    net:add(nn.SpatialConvolution(n, 3 * (2 + 4), 1, 1))  -- aspect ratios { 1:1, 2:1, 1:2 } x { class, left, top, width, height }
    return net
  end

  
    
  local conv_outputs = {}
  
  --#######################
  -- reduce feature map dimension
  local net= nn.Sequential()

  -- conv input is n*fea_dim*s*s
  if cfg.flow == 2 then
    ConvPoolBlock(net, 1024, layers[1].filters, layers[1].kW, layers[1].kH, layers[1].padW, layers[1].padH, layers[1].dropout, layers[1].conv_steps,1) 
  else
    ConvPoolBlock(net, 512, layers[1].filters, layers[1].kW, layers[1].kH, layers[1].padW, layers[1].padH, layers[1].dropout, layers[1].conv_steps,1) 
  end
  -- two path -> enc and dec
  local concat = nn.ConcatTable()

  -- to enc, after split conv is a table of n tensors with size fea_dim*s*s
  concat:add(nn.SplitTable(1)) 
  -- to dec, reverse the tensors along dim 1
  -- after split conv is a table of n tensors with size fea_dim*s*s
  local reverse = nn.Sequential()
  reverse:add(nn.SeqReverseSequence(1))
  reverse:add(nn.SplitTable(1))

  concat:add(reverse)
  -- output of the net are two tables, n tensors in each table
  net:add(concat) 


  --###########################
  -- Hjj_encoder_decoder_ConvGRU.lua
  -- enc is nn.Sequential->((nn.Sequencer->ConvGRU)->SelectTable(-1)) 
  
  local enc = create_encoder(layers[2])
  -- dec is nn.Sequential->((nn.Sequencer->ConvGRU)->(nn.Sequencer->Replicate)->JoinTable)
  local dec = create_decoder(layers[3])
  local input = - nn.Sequencer(nn.Identity())
  local dec_output = input - dec
  
  table.insert(conv_outputs, dec_output)



  local proposal_outputs = {}
  for i,a in ipairs(anchor_nets) do
    table.insert(proposal_outputs, AnchorNetwork(layers[a.input].filters, a.n, a.kW)(conv_outputs[a.input])) -- a.input == 1
  end
  table.insert(proposal_outputs, conv_outputs[#conv_outputs])
  
    -- create proposal net module, outputs: anchor net outputs followed by last conv-layer output
    -- dec is rpn.modules[2]
  local rpn = nn.gModule({ input }, proposal_outputs)
  rpn.convgru = dec.convgru
  
  local function init(module, name)
    local function init_module(m)
      for k,v in pairs(m:findModules(name)) do
        local n = v.kW * v.kH * v.nOutputPlane
        v.weight:normal(0, math.sqrt(2 / n))
        v.bias:zero()
      end
    end
    module:apply(init_module)
  end

  init(rpn, 'nn.SpatialConvolution')
  init(net, 'nn.SpatialConvolution')
  
  return {red_net = net, enc = enc, rpn = rpn}
end

function create_classification_net(inputs, class_count, class_layers)
  -- create classifiaction network
  local net = nn.Sequential()
  local prev_input_count = inputs
  for i,l in ipairs(class_layers) do
    net:add(nn.Linear(prev_input_count, l.n)) 
    if l.batch_norm then
      net:add(nn.BatchNormalization(l.n))
    end
    net:add(nn.PReLU()) 
    if l.dropout and l.dropout > 0 then
      net:add(nn.Dropout(l.dropout))
    end
    prev_input_count = l.n
  end
  net =  nn.Sequencer(net)
  
  local input =  nn.Identity()() 
  local node = net(input) 
  
  -- now the network splits into regression and classification branches
  
  -- regression output
  --local tmp_linear = nn.Linear(prev_input_count,4)
  local tmp_linear = nn.Linear(prev_input_count,4*class_count)
  tmp_linear = nn.Sequencer(tmp_linear)
  local rout =  tmp_linear(node) 
  
  -- classification output
  local cnet = nn.Sequential()
  cnet:add(nn.Linear(prev_input_count, class_count)) 
  cnet:add(nn.LogSoftMax())
  
  cnet = nn.Sequencer(cnet)
  local cout =  cnet(node)
  
  -- create bbox finetuning + classification output
  local model = nn.gModule({ input }, { rout, cout })

  local function init(module, name)
    local function init_module(m)
      for k,v in pairs(m:findModules(name)) do
        local n = v.kW * v.kH * v.nOutputPlane
        v.weight:normal(0, math.sqrt(2 / n))
        v.bias:zero()
      end
    end
    module:apply(init_module)
  end

  init(model, 'nn.SpatialConvolution')
  return model
end

function create_model(cfg, layers, anchor_nets, class_layers)
  local cnet_ninputs = cfg.roi_pooling.kh * cfg.roi_pooling.kw * layers[#layers].filters
  local model = 
  {
    cfg = cfg,
    layers = layers,
    pnet = create_proposal_net(layers, anchor_nets, cfg),
    cnet=create_classification_net(cnet_ninputs, cfg.class_count + 1, class_layers)
  }
  return model
end
