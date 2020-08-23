#ifdef __cplusplus
extern "C" {
#endif

#define _USE_MATH_DEFINES
    
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "fan_projection_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)
    
struct Point{
   float x;
   float y;
};

__device__ Point rotate_point(Point pos, float angle)
{
    //Rotation by about the origin
    Point res;
    res.x = pos.x * cos(-angle) - pos.y * sin(-angle);
    res.y = pos.x * sin(-angle) + pos.y * cos(-angle);
    return res;  
}        
    
__device__ Point to_img_coord(Point pos, float pixel_spacing){
    Point res;
    
    res.x = pos.x / pixel_spacing + 255.5;
    res.y = 255.5 - pos.y / pixel_spacing;
    
    return res;
}        
    
__device__ float bilinear_interpolate(const float* bottom_data,
    const int height, const int width,
    float y, float x,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int) y;
  int x_low = (int) x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (float) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (float) x_low;
  } else {
    x_high = x_low + 1;
  }

  float ly = y - y_low;
  float lx = x - x_low;
  float hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  float v1 = bottom_data[y_low * width + x_low];
  float v2 = bottom_data[y_low * width + x_high];
  float v3 = bottom_data[y_high * width + x_low];
  float v4 = bottom_data[y_high * width + x_high];
  float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}
  
  
__global__ void ProjectionForward(const int nthreads, const float* bottom_data,
    const int height, const int width,
    const int channels, const int output_height, const int output_width,
    float* top_data, int a)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        //Hardcode Config
        float M = 1024;
        float N = 720;
        float DET_LEN = 102.4;
        float DET_X = 40.8;
        float DET_SPACING = DET_LEN / M;
        float pixel_spacing = 36.0 / 512.;
        Point source_pos = {-54.1, 0};
 
        // (n, c, ch, cw) is an element in the output
        int n = index;
        int cw = n % output_width;
        n /= output_width;
        int ch = n % output_height;
        n /= output_height;
        int c = n % channels;
        n /= channels;
                
        const float* offset_bottom_data = bottom_data + (n * channels + c) * height * width;
        
        int theta_idx = ch;
        int detector_idx = cw;
        
        float ang = theta_idx * 2.* M_PI / N;
        Point detector_pos = {DET_X, -((M/2. -1)*DET_SPACING + DET_SPACING/2. - DET_SPACING*detector_idx)};

        Point rotated_source = to_img_coord(rotate_point(source_pos, ang), pixel_spacing);
        Point rotated_detector = to_img_coord(rotate_point(detector_pos, ang), pixel_spacing);

        //start from source
        float distance_x = rotated_detector.x - rotated_source.x;
        float distance_y = rotated_detector.y - rotated_source.y;
        float distance = sqrt(distance_x*distance_x + distance_y*distance_y);
           
        
        float output_val = 0.;
        for (int i = 360; i < int(distance) - 120; i++)
        {
            float x = rotated_source.x + i / distance * distance_x;
            float y = rotated_source.y + i / distance * distance_y;
            
            float val = bilinear_interpolate(offset_bottom_data, height, width, y, x, index);
            output_val += val;
        }
            
        top_data[index] = output_val*pixel_spacing;
    }
}


int ProjectionForwardLaucher(
    const float* bottom_data, const int batch_size, const int height,
    const int width, const int channels, const int output_height,
    const int output_width, float* top_data, cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    const int output_size = batch_size * output_height * output_width * channels;
    cudaError_t err;


    ProjectionForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, bottom_data, height, width, channels, output_height,
      output_width, top_data, output_size);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

__device__ void bilinear_interpolate_gradient(
    const int height, const int width,
    float y, float x,
    float & w1, float & w2, float & w3, float & w4,
    int & x_low, int & x_high, int & y_low, int & y_high,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int) y;
  x_low = (int) x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (float) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (float) x_low;
  } else {
    x_high = x_low + 1;
  }

  float ly = y - y_low;
  float lx = x - x_low;
  float hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}    
    

__global__ void ProjectionBackward(const int nthreads, const float* top_diff,
    const int height, const int width, const int channels,
    const int output_height, const int output_width, float* bottom_diff) 
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        //Hardcode Config
        float M = 1024;
        float N = 720;
        float DET_LEN = 102.4;
        float DET_X = 40.8;
        float DET_SPACING = DET_LEN / M;
        float pixel_spacing = 36.0 / 512.;
        Point source_pos = {-54.1, 0};

        // (n, c, ch, cw) is an element in the output
        int n = index;
        int cw = n % output_width;
        n /= output_width;
        int ch = n % output_height;
        n /= output_height;
        int c = n % channels;
        n /= channels;
                
        float* offset_bottom_diff = bottom_diff + (n * channels + c) * height * width;
        int top_offset = (n * channels + c) * output_height * output_width;
        const float* offset_top_diff = top_diff + top_offset;
        const float top_diff_this_pos = offset_top_diff[ch * output_width + cw];
        
        int theta_idx = ch;
        int detector_idx = cw;
        
        float ang = theta_idx * 2.* M_PI / N;
        Point detector_pos = {DET_X, -((M/2. -1)*DET_SPACING + DET_SPACING/2. - DET_SPACING*detector_idx)};

        Point rotated_source = to_img_coord(rotate_point(source_pos, ang), pixel_spacing);
        Point rotated_detector = to_img_coord(rotate_point(detector_pos, ang), pixel_spacing);

        //start from source
        float distance_x = rotated_detector.x - rotated_source.x;
        float distance_y = rotated_detector.y - rotated_source.y;
        float distance = sqrt(distance_x*distance_x + distance_y*distance_y);
           
        for (int i = 360; i < int(distance) - 120; i++)
        {
            float x = rotated_source.x + i / distance * distance_x;
            float y = rotated_source.y + i / distance * distance_y;
            
            float w1, w2, w3, w4;
            int x_low, x_high, y_low, y_high;
            
            bilinear_interpolate_gradient(height, width, y, x,
                w1, w2, w3, w4,
                x_low, x_high, y_low, y_high,
                index);
            
            float g1 = top_diff_this_pos * w1;
            float g2 = top_diff_this_pos * w2;
            float g3 = top_diff_this_pos * w3;
            float g4 = top_diff_this_pos * w4; 
            
            if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0)
            {
              atomicAdd(offset_bottom_diff + y_low * width + x_low, static_cast<float>(g1));
              atomicAdd(offset_bottom_diff + y_low * width + x_high, static_cast<float>(g2));
              atomicAdd(offset_bottom_diff + y_high * width + x_low, static_cast<float>(g3));
              atomicAdd(offset_bottom_diff + y_high * width + x_high, static_cast<float>(g4));
            } // if
        }
        
//         for (int t = 0; t < M; t++)
//         {
            
//             float beta = t * 0.5 * M_PI / 180 - M_PI;
//             float th = M_PI/2.0 + beta + res.phi;
//             float s = D * res.rho * sin(th) / (D + res.rho * cos(th));
//             float u = pow((1.0 + res.rho * sin(beta - res.phi) / D), 2);
            
//             float x = ((s - nT)/ T + 0.5);   
//             float y = float(t);
            
//             float w1, w2, w3, w4;
//             int x_low, x_high, y_low, y_high;
            
//             bilinear_interpolate_gradient(height, width, y, x,
//                 w1, w2, w3, w4,
//                 x_low, x_high, y_low, y_high,
//                 index);
            
//             float g1 = top_diff_this_pos * w1 / u;
//             float g2 = top_diff_this_pos * w2 / u;
//             float g3 = top_diff_this_pos * w3 / u;
//             float g4 = top_diff_this_pos * w4 / u; 
            
//             if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0)
//             {
//               atomicAdd(offset_bottom_diff + y_low * width + x_low, static_cast<float>(g1));
//               atomicAdd(offset_bottom_diff + y_low * width + x_high, static_cast<float>(g2));
//               atomicAdd(offset_bottom_diff + y_high * width + x_low, static_cast<float>(g3));
//               atomicAdd(offset_bottom_diff + y_high * width + x_high, static_cast<float>(g4));
//             } // if
//           }
  }
}

int ProjectionBackwardLaucher(const float* top_diff, const int batch_size,
    const int height, const int width, const int channels, const int output_height,
    const int output_width, float* bottom_diff, cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    const int output_size = batch_size * output_height * output_width * channels;
    cudaError_t err;

    ProjectionBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, top_diff, height, width, channels, output_height,
      output_width, bottom_diff);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


#ifdef __cplusplus
}
#endif