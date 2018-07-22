-- global variant
-- include vgg and reduce_net
feature_extraction_net_layers = {
  {kW=3 , kH=3 , dW=1 , dH=1 , padW=1 , padH=1 }, -- conv1_1
  {kW=2 , kH=2 , dW=2 , dH=2 , padW=0 , padH=0 }, -- pooling
  {kW=3 , kH=3 , dW=1 , dH=1 , padW=1 , padH=1 }, -- conv2_1
  {kW=3 , kH=3 , dW=1 , dH=1 , padW=1 , padH=1 }, -- conv2_2
  {kW=2 , kH=2 , dW=2 , dH=2 , padW=0 , padH=0 }, -- pooling
  {kW=3 , kH=3 , dW=1 , dH=1 , padW=1 , padH=1 }, -- conv3_1
  {kW=3 , kH=3 , dW=1 , dH=1 , padW=1 , padH=1 }, -- conv3_2
  {kW=3 , kH=3 , dW=1 , dH=1 , padW=1 , padH=1 }, -- conv3_3
  {kW=2 , kH=2 , dW=2 , dH=2 , padW=0 , padH=0 }, -- pooling
  {kW=3 , kH=3 , dW=1 , dH=1 , padW=1 , padH=1 }, -- conv4_1
  {kW=3 , kH=3 , dW=1 , dH=1 , padW=1 , padH=1 }, -- conv4_2
  {kW=3 , kH=3 , dW=1 , dH=1 , padW=1 , padH=1 }, -- conv4_3
  {kW=2 , kH=2 , dW=2 , dH=2 , padW=0 , padH=0 }, -- pooling
  {kW=3 , kH=3 , dW=1 , dH=1 , padW=1 , padH=1 }, -- conv5_1
  {kW=3 , kH=3 , dW=1 , dH=1 , padW=1 , padH=1 }, -- conv5_2
  {kW=3 , kH=3 , dW=1 , dH=1 , padW=1 , padH=1 } -- conv5_3
}

-- lenth of GRU
ConvGRU_rho = 8	

GPU_ID = 1
DEFAULT_GPU=1

local class_count = 22

Det_Table = torch.Tensor(class_count):fill(0)
Bg_Table = torch.Tensor(class_count):fill(0)
Pos_Prop_Table = torch.Tensor(class_count):fill(0)
Neg_Prop_Table = torch.Tensor(class_count):fill(0)