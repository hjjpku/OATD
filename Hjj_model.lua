require 'Hjj_model_utilities'

function en_de_model(cfg)
  -- first layer is reduc_net which is to reduce dimension from 512->256
  -- second layer is in the encoder Conv_GRU
  -- third layer is in the decoder Conv_GRU, do RoIs pooling on this layer
  local filter_dim = 128
  local layers = { 
    { filters=filter_dim, kW=1, kH=1, padW=0, padH=0, dropout=0.3, conv_steps=1 },
    { filters=filter_dim, kW=3, kH=3, padW=1, padH=1, dropout=0, conv_steps=1 },
    { filters=filter_dim, kW=3, kH=3, padW=1, padH=1, dropout=0, conv_steps=1 }
  }
  
  local anchor_nets = {
    { kW=2, n=filter_dim, input=1 },   -- input refers to the last 'layer' defined above
    { kW=3, n=filter_dim, input=1 },
    { kW=5, n=filter_dim, input=1 },
    { kW=7, n=filter_dim, input=1 }
  }
  
  local class_layers =  {
    { n=4096, dropout=0.5, batch_norm=false },
    { n=4096, dropout=0.5 },
  }
  
  return create_model(cfg, layers, anchor_nets, class_layers) --Hjj_model_utilities.lua
end

return en_de_model