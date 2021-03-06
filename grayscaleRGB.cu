/*
	RGB GREYSCALE
	HUDSON PEDEN
	10/18/2016
*/

#include "libwb\wb.h"
#include "cuda.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
		    }                                                                     \
      } while (0)



#define TILE_WIDTH 3
#define TILE_HEIGHT 1
#define OUTCHANNELS 3 //(Color:3 ; Grayscale: 1)

__global__ void rgb2gray(float *grayImage, float *rgbImage, int channels, int width, int height) {
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	if (x < width  && y < height){
		int offset = y * width + x;
		__shared__ float greyVal;
		
		greyVal = 0;
		syncthreads();

		//int colorSel = offset % 3;  // R=0 G=1 B=2


		greyVal += rgbImage[offset];
		


			//(rgbImage[offset + 0 - colorSel] + rgbImage[offset + 1 - colorSel] + rgbImage[offset + 2 - colorSel]) / 3.0;
			//grayImage[offset] = greyVal;		// RED
			//grayImage[offset + 1] = greyVal;	// GREEN
			//grayImage[offset + 2] = greyVal;	// BLUE
		


		/*
			WAIT FOR RED THREADS TO COMPUTE USING BARRIER SYNC
		*/
		__syncthreads(); 

		/*
			APPLY MODIFICATIONS
		*/
		grayImage[offset] = greyVal / 3.0;

		/*
		
		else if (offset % 3 == 1) greyVal = (rgbImage[offset - 1] + rgbImage[offset + 0] + rgbImage[offset + 1]) / 3.0;

		

		else greyVal = (rgbImage[offset - 2] + rgbImage[offset - 1] + rgbImage[offset + 0]) / 3.0;
		*/
		

		
	}

}

int main(int argc, char *argv[]) {
	wbArg_t args;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *deviceInputImageData;
	float *deviceOutputImageData;

	args = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(args, 0);

	inputImage = wbImport(inputImageFile); //"pict.ppm"

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	// For this lab the value is always 3
	imageChannels = wbImage_getChannels(inputImage);

	// Since the image is monochromatic, it only contains one channel
	outputImage = wbImage_new(imageWidth, imageHeight, OUTCHANNELS);// 3);// 1);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * OUTCHANNELS * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Compute, "Doing the computation on the GPU");

	/*
		CALL KERNEL FUNCTION
	*/

	dim3 dimGrid(ceil((float)(imageWidth * OUTCHANNELS) / TILE_WIDTH), ceil((float)imageHeight / TILE_HEIGHT));
	dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT, 1);
	rgb2gray <<<dimGrid, dimBlock >>>(deviceOutputImageData, deviceInputImageData, imageChannels, imageWidth * OUTCHANNELS, imageHeight);
	
	
	wbTime_stop(Compute, "Doing the computation on the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * OUTCHANNELS * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");
	//	for (int i = 0; i < 100; i++)fprintf(stderr, "%2.1f ", hostInputImageData[i]); fprintf(stderr, "\n");
	//	for (int i = 0; i < 100; i++)fprintf(stderr, "%2.1f ", hostOutputImageData[i]); fprintf(stderr, "\n");
	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(args, outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}
