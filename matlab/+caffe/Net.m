classdef Net < handle
  % Wrapper class of caffe::Net in matlab
  
  properties (Access = private)
    hNet_self
    attributes
    % attribute fields
    %     hLayer_layers
    %     hBlob_blobs
    %     input_blob_indices
    %     output_blob_indices
    %     layer_names
    %     blob_names
  end
  properties (SetAccess = private)
    layer_vec
    blob_vec
    inputs
    outputs
    name2layer_index
    name2blob_index
    layer_names
    blob_names
  end
  
  methods
    function self = Net(varargin)
      % decide whether to construct a net from model_file or handle
      if ~(nargin == 1 && isstruct(varargin{1}))
        % construct a net from model_file
        self = caffe.get_net(varargin{:});
        return
      end
      % construct a net from handle
      hNet_net = varargin{1};
      CHECK(is_valid_handle(hNet_net), 'invalid Net handle');
      
      % setup self handle and attributes
      self.hNet_self = hNet_net;
      self.attributes = caffe_('net_get_attr', self.hNet_self);
      
      % setup layer_vec
      self.layer_vec = caffe.Layer.empty();
      for n = 1:length(self.attributes.hLayer_layers)
        self.layer_vec(n) = caffe.Layer(self.attributes.hLayer_layers(n));
      end
      
      % setup blob_vec
      self.blob_vec = caffe.Blob.empty();
      for n = 1:length(self.attributes.hBlob_blobs);
        self.blob_vec(n) = caffe.Blob(self.attributes.hBlob_blobs(n));
      end
      
      % setup input and output blob and their names
      % note: add 1 to indices as matlab is 1-indexed while C++ is 0-indexed
      self.inputs = ...
        self.attributes.blob_names(self.attributes.input_blob_indices + 1);
      self.outputs = ...
        self.attributes.blob_names(self.attributes.output_blob_indices + 1);
      
      % create map objects to map from name to layers and blobs
      self.name2layer_index = containers.Map(self.attributes.layer_names, ...
        1:length(self.attributes.layer_names));
      self.name2blob_index = containers.Map(self.attributes.blob_names, ...
        1:length(self.attributes.blob_names));
      
      % expose layer_names and blob_names for public read access
      self.layer_names = self.attributes.layer_names;
      self.blob_names = self.attributes.blob_names;
    end
    function layer = layers(self, layer_name)
      CHECK(ischar(layer_name), 'layer_name must be a string');
      layer = self.layer_vec(self.name2layer_index(layer_name));
    end
    function blob = blobs(self, blob_name)
      CHECK(ischar(blob_name), 'blob_name must be a string');
      blob = self.blob_vec(self.name2blob_index(blob_name));
    end
    function blob = params(self, layer_name, blob_index)
      CHECK(ischar(layer_name), 'layer_name must be a string');
      CHECK(isscalar(blob_index), 'blob_index must be a scalar');
      blob = self.layer_vec(self.name2layer_index(layer_name)).params(blob_index);
    end
    function set_input_arrays(self, data,labels, layer_idx)
      caffe_('net_set_input_arrays', self.hNet_self, data, labels, layer_idx);
    end
    function array = get_input_arrays(self, layer_idx)
      array = caffe_('net_get_input_arrays', self.hNet_self, layer_idx);
    end
    function forward_prefilled(self)
      caffe_('net_forward', self.hNet_self);
    end
    function forward_from_to(self,from,to)
      caffe_('net_forward', self.hNet_self,int32(from),int32(to));
    end
    function backward_prefilled(self)
      caffe_('net_backward', self.hNet_self);
    end
    function forward_res=forward_batch(self,output_hBlob,n_samps)
      forward_res=caffe_('net_forward_batch', self.hNet_self, output_hBlob.get_handle(), int32(n_samps));
    end
    function [forward_res,backward_res]=forward_backward_batch(self,input_hBlob,output_hBlob,n_samps)
      [forward_res,backward_res]=caffe_('net_forward_backward_batch', self.hNet_self, input_hBlob.get_handle(), output_hBlob.get_handle(), int32(n_samps));
    end
    function [jacobian1,jacobian2]=get_jacobian(self,layer_idx,blob_obj)
      [jacobian1,jacobian2] = caffe_('net_get_jacobian', self.hNet_self,int32(layer_idx),int32(blob_obj));
    end
    function loss_diff=get_loss_diff(self,layer_idx)
      [loss_diff] = caffe_('net_get_loss_diff', self.hNet_self,int32(layer_idx));
    end
    function lr_mults=net_get_lr_mults(self)
      [lr_mults] = caffe_('net_get_lr_mults', self.hNet_self);
    end
    function net_set_lr_mults(self,lr_mults)
      caffe_('net_set_lr_mults', self.hNet_self, lr_mults);
    end
    function res = forward(self, input_data)
      CHECK(iscell(input_data), 'input_data must be a cell array');
      CHECK(length(input_data) == length(self.inputs), ...
        'input data cell length must match input blob number');
      % copy data to input blobs
      for n = 1:length(self.inputs)
        self.blobs(self.inputs{n}).set_data(input_data{n});
      end
      self.forward_prefilled();
      % retrieve data from output blobs
      res = cell(length(self.outputs), 1);
      for n = 1:length(self.outputs)
        res{n} = self.blobs(self.outputs{n}).get_data();
      end
    end
    function res = backward(self, output_diff)
      CHECK(iscell(output_diff), 'output_diff must be a cell array');
      CHECK(length(output_diff) == length(self.outputs), ...
        'output diff cell length must match output blob number');
      % copy diff to output blobs
      for n = 1:length(self.outputs)
        self.blobs(self.outputs{n}).set_diff(output_diff{n});
      end
      self.backward_prefilled();
      % retrieve diff from input blobs
      res = cell(length(self.inputs), 1);
      for n = 1:length(self.inputs)
        res{n} = self.blobs(self.inputs{n}).get_diff();
      end
    end
    function copy_from(self, weights_file)
      CHECK(ischar(weights_file), 'weights_file must be a string');
      CHECK_FILE_EXIST(weights_file);
      caffe_('net_copy_from', self.hNet_self, weights_file);
    end
    function reshape(self)
      caffe_('net_reshape', self.hNet_self);
    end
    function save(self, weights_file)
      CHECK(ischar(weights_file), 'weights_file must be a string');
      caffe_('net_save', self.hNet_self, weights_file);
    end
  end
end
