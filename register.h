#include <caffe\common.hpp>
#include <caffe\layers\input_layer.hpp>
#include <caffe\layers\inner_product_layer.hpp>
#include <caffe\layers\flatten_layer.hpp>

namespace caffe
{
	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(FlattenLayer);
}