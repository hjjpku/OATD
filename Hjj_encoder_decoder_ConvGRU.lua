require 'rnn'
require 'Hjj_ConvGRU'
require 'Hjj_global'
require 'nn'
require 'nngraph'


--initialization from MSR
local function MSRinit(net)
    local function init(name)
        for k,v in pairs(net:findModules(name)) do
            local n = v.kW*v.kH*v.nOutputPlane
            v.weight:normal(0,math.sqrt(2/n))
            v.bias:zero()
        end
    end
    -- have to do for both backends
    init'cudnn.SpatialConvolution'
    init'nn.SpatialConvolution'
end

-- forward coupling: copy encoder's outputs to decoder
function forwardConnection(enc, dec)
    dec.convgru.userPrevOutput = nn.rnn.recursiveCopy(dec.convgru.userPrevOutput,enc.convgru.outputs[ConvGRU_rho])
end

-- backward coupling: copy decoder's gradients to encoder
function backwardConnection(enc, dec)
    enc.convgru:setGradHiddenState(ConvGRU_rho, dec.convgru:getGradHiddenState(0))
end

-- Encoder
function create_encoder(layers)
    local enc = nn.Sequential()
    enc.convgru = nn.ConvGRU(layers.filters, layers.filters, ConvGRU_rho, layers.kW, layers.kH,  layers.padW, layers.padH)
    enc:add(nn.Sequencer(enc.convgru))
    enc:add(nn.SelectTable(-1)) --output is tensor with size fea_dim*s*s
    MSRinit(enc)
    return enc  
end


-- Decoder
function create_decoder(layers)
    local dec = nn.Sequential()
    dec.convgru  = nn.ConvGRU(layers.filters, layers.filters, ConvGRU_rho, layers.kW, layers.kH,  layers.padW, layers.padH)
    dec:add(nn.Sequencer(dec.convgru))
    dec:add(nn.Sequencer(nn.Replicate(1)))
    dec:add(nn.JoinTable(1)) --output is tensor with size n*fea_dim*s*s
    MSRinit(dec)
    return dec  
end



