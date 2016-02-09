#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

template <typename Dtype>
int EuclideanLossLayer<Dtype>::Get_diff_shape(int idx){
  return (&diff_)->shape(idx);
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Get_diff(Blob<Dtype>* tmp_diff_blob) {

  Dtype* tmp_diff_mem_ptr = NULL;
  Dtype* internal_diff_mem_ptr = NULL;

  switch (Caffe::mode()) {
  case Caffe::CPU:
    tmp_diff_mem_ptr = tmp_diff_blob->mutable_cpu_diff();
    internal_diff_mem_ptr = (&diff_)->mutable_cpu_diff();
    break;
  case Caffe::GPU:
    tmp_diff_mem_ptr = tmp_diff_blob->mutable_gpu_diff();
    internal_diff_mem_ptr = (&diff_)->mutable_gpu_diff();
    break;
  }
  caffe_copy((&diff_)->count(), internal_diff_mem_ptr, tmp_diff_mem_ptr);
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Set_diff(Blob<Dtype>* diff_blob_ptr) {

  Blob<Dtype>* internal_diff_blob_ptr=&diff_;

  Dtype* diff_blob_mem_ptr = NULL;
  Dtype* internal_diff_blob_mem_ptr_ = NULL;
  
  switch (Caffe::mode()) {
  case Caffe::CPU:
    diff_blob_mem_ptr = diff_blob_ptr->mutable_cpu_diff();
    internal_diff_blob_mem_ptr_ = internal_diff_blob_ptr->mutable_cpu_diff();
    break;
  case Caffe::GPU:
    diff_blob_mem_ptr = diff_blob_ptr->mutable_gpu_diff();
    internal_diff_blob_mem_ptr_ = internal_diff_blob_ptr->mutable_gpu_diff();
    break;
  }

  caffe_copy(internal_diff_blob_ptr->count(), diff_blob_mem_ptr, internal_diff_blob_mem_ptr_);

}


#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
