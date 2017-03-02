
#include <algorithm>
#include <vector>

#include "caffe/layers/triplet_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
 
namespace caffe {
 
template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(
 const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top) {

 LossLayer<Dtype>::LayerSetUp(bottom, top);
 top[0]->Reshape(1,1,1,1);//loss

 margin = this->layer_param_.triplet_loss_param().margin();//读取距离参数a 默认为1
 // bottom[0] : f(a); bottom[1] : f(p); bottom[2] : f(n)
 //下面2行检查三元组的batch是否一样
 CHECK_EQ(bottom[0]->num(), bottom[1]->num());
 //下面2行检查三元组的通道数是否一样
 
 //下面6行检查三元组的h、w是否为1
 CHECK_EQ(bottom[0]->height(), 1);
 CHECK_EQ(bottom[0]->width(), 1);
 CHECK_EQ(bottom[1]->height(), 1);
 CHECK_EQ(bottom[1]->width(), 1);

 
 diff_ap_.Reshape(1, bottom[0]->channels(), 1, 1);
 diff_an_.Reshape(1, bottom[0]->channels(), 1, 1);
 diff_pn_.Reshape(1, bottom[0]->channels(), 1, 1);



 dist_sq_ap_.Reshape(1, 1, 1, 1);
 dist_sq_an_.Reshape(1, 1, 1, 1);

}
 
template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>&bottom,
    const vector<Blob<Dtype>*>&top) {

  Blob<Dtype>* data_ =bottom[0];
  Blob<Dtype>* label_ = bottom[1];

  int nums =bottom[0]->num();
  int channels = bottom[0]->channels();

  Dtype loss(0.0);
  count=Dtype(0.0);//count of truiplet in this batch
  for(int i=0;i<nums;++i)
  {
    //std::cout<<label_b->cpu_data()[i]<<std::endl;
    for(int j=i+1;j<nums;++j)
    {
      if(label_->cpu_data()[i]==label_->cpu_data()[j])
      {
        for(int k=j+1;k<nums;++k)
        {
          if(label_->cpu_data()[i]!=label_->cpu_data()[k])
          {
            caffe_sub(channels,
              data_->cpu_data()+(i*channels),data_->cpu_data()+(j*channels),
              diff_ap_.mutable_cpu_data()); 

            dist_sq_ap_.mutable_cpu_data()[0]=caffe_cpu_dot(channels,
              diff_ap_.cpu_data(),diff_ap_.cpu_data());

            caffe_sub(channels,
              data_->cpu_data()+(i*channels),data_->cpu_data()+(k*channels),
              diff_an_.mutable_cpu_data()); 

            dist_sq_an_.mutable_cpu_data()[0]=caffe_cpu_dot(channels,
              diff_an_.cpu_data(),diff_an_.cpu_data());  
            //std::cout<<dist_sq_ap_.cpu_data()[0]<<"    "<<dist_sq_an_.cpu_data()[0]<<std::endl;
            Dtype mdist = std::max(margin +dist_sq_ap_.cpu_data()[0] - dist_sq_an_.cpu_data()[0], Dtype(0.0));   
            loss += mdist; 
            count ++;
          }
        }
      }
    }
      
  }


  
  loss = loss / static_cast<Dtype>(count);
  top[0]->mutable_cpu_data()[0] = loss;//计算loss
}
 
template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>&propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //Dtype margin =this->layer_param_.triplet_loss_param().margin();
  //const Dtype* sampleW = bottom[3]->cpu_data();//权值？？

  Blob<Dtype>* data_ =bottom[0];
  Blob<Dtype>* label_ = bottom[1];

 //std::cout<<count<<std::endl;

  for(int i=0;i< data_->count();i++)
  {
   data_->mutable_cpu_diff()[i]=Dtype(0.0);
  }

  int nums =bottom[0]->num();
  int channels = bottom[0]->channels();

  const Dtype alpha = top[0]->cpu_diff()[0]/static_cast<Dtype>(count);

  for(int i=0;i<nums;++i)
  {
    for(int j=i+1;j<nums;++j)
    {
      if(label_->cpu_data()[i]==label_->cpu_data()[j])
      {
        for(int k=j+1;k<nums;++k)
        {
          if(label_->cpu_data()[i]!=label_->cpu_data()[k])
          {
            caffe_sub(channels,
              data_->cpu_data()+(i*channels),data_->cpu_data()+(j*channels),
              diff_ap_.mutable_cpu_data()); 

            dist_sq_ap_.mutable_cpu_data()[0]=caffe_cpu_dot(channels,
              diff_ap_.cpu_data(),diff_ap_.cpu_data());

            caffe_sub(channels,
              data_->cpu_data()+(i*channels),data_->cpu_data()+(k*channels),
              diff_an_.mutable_cpu_data()); 

            dist_sq_an_.mutable_cpu_data()[0]=caffe_cpu_dot(channels,
              diff_an_.cpu_data(),diff_an_.cpu_data());  

            caffe_sub(channels,
              data_->cpu_data()+(j*channels),data_->cpu_data()+(k*channels),
              diff_pn_.mutable_cpu_data()); 

            Dtype mdist = std::max(margin +dist_sq_ap_.cpu_data()[0] - dist_sq_an_.cpu_data()[0], Dtype(0.0));  
            //std::cout<<dist_sq_ap_.cpu_data()[0]<<"    "<<dist_sq_an_.cpu_data()[0]<<std::endl;
            //std::cout<<mdist<<std::endl;
            
            if(mdist!=0)
            {
  
                caffe_axpy(channels,Dtype(-2.0)*alpha,diff_pn_.cpu_data(),data_->mutable_cpu_diff()+(i*channels));

                caffe_axpy(channels,Dtype(-2.0)*alpha,diff_ap_.cpu_data(),data_->mutable_cpu_diff()+(j*channels));

                caffe_axpy(channels,Dtype(2.0)*alpha,diff_an_.cpu_data(),data_->mutable_cpu_diff()+(k*channels));

            }
            else
            {
                caffe_axpy(channels,Dtype(0.0),diff_pn_.cpu_data(),data_->mutable_cpu_diff()+(i*channels));

                caffe_axpy(channels,Dtype(0.0),diff_ap_.cpu_data(),data_->mutable_cpu_diff()+(j*channels));

                caffe_axpy(channels,Dtype(0.0),diff_an_.cpu_data(),data_->mutable_cpu_diff()+(k*channels));

            }
          }
        }
      }
    }
  }
}
 
#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif
 
INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);
 
} // namespace caffe
