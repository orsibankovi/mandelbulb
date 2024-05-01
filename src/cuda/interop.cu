#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "interop.cuh"
#include "stdio.h"
#include <memory>
#include <cuda/std/complex>

#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}

uint32_t imageWidth, imageHeight;
float _zoom = 1.0f;
cudaExternalMemory_t cudaExtMemImageBuffer; // memory handler to the imported memory allocation
cudaMipmappedArray_t cudaMipmappedImageArray; // the image interpreted as a mipmapped array
cudaSurfaceObject_t surfaceObject; // surface object to the first mip level of the array. Allows write

void freeExportedVulkanImage()
{
    checkCudaError(cudaDestroySurfaceObject(surfaceObject));
    checkCudaError(cudaFreeMipmappedArray(cudaMipmappedImageArray));
    checkCudaError(cudaDestroyExternalMemory(cudaExtMemImageBuffer));
}

void exportVulkanImageToCuda_R8G8B8A8Unorm(void* mem, VkDeviceSize size, VkDeviceSize offset, uint32_t width, uint32_t height)
{
    imageWidth = width;
    imageHeight = height;

    // import memory into cuda through native handle (win32)
    cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
    memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
    cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32; // after win8
    cudaExtMemHandleDesc.handle.win32.handle = mem; // allocation handle
    cudaExtMemHandleDesc.size = size; // allocation size
   
    checkCudaError(cudaImportExternalMemory(&cudaExtMemImageBuffer, &cudaExtMemHandleDesc));

    // extract mipmapped array from memory
    cudaExternalMemoryMipmappedArrayDesc externalMemoryMipmappedArrayDesc;
    memset(&externalMemoryMipmappedArrayDesc, 0, sizeof(externalMemoryMipmappedArrayDesc));

    // we want ot interpret the raw memory as an image so we need to specify its format and layout
    cudaExtent extent = make_cudaExtent(width, height, 0);
    cudaChannelFormatDesc formatDesc; // 4 channel, 8 bit per channel, unsigned
    formatDesc.x = 8;
    formatDesc.y = 8;
    formatDesc.z = 8;
    formatDesc.w = 8;
    formatDesc.f = cudaChannelFormatKindUnsigned;

    externalMemoryMipmappedArrayDesc.offset = offset; // the image starts here
    externalMemoryMipmappedArrayDesc.formatDesc = formatDesc;
    externalMemoryMipmappedArrayDesc.extent = extent;
    externalMemoryMipmappedArrayDesc.flags = 0;
    externalMemoryMipmappedArrayDesc.numLevels = 1; // no mipmapping
    checkCudaError(cudaExternalMemoryGetMappedMipmappedArray(&cudaMipmappedImageArray, cudaExtMemImageBuffer, &externalMemoryMipmappedArrayDesc));

    // extract first level
    cudaArray_t cudaMipLevelArray;
    checkCudaError(cudaGetMipmappedArrayLevel(&cudaMipLevelArray, cudaMipmappedImageArray, 0));

    // create surface object for writing
    cudaResourceDesc resourceDesc;
    memset(&resourceDesc, 0, sizeof(resourceDesc));
    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = cudaMipLevelArray;
    
    checkCudaError(cudaCreateSurfaceObject(&surfaceObject, &resourceDesc));
}

cudaExternalSemaphore_t cudaWaitsForVulkanSemaphore, vulkanWaitsForCudaSemaphore;

void freeExportedSemaphores()
{
    checkCudaError(cudaDestroyExternalSemaphore(cudaWaitsForVulkanSemaphore));
    checkCudaError(cudaDestroyExternalSemaphore(vulkanWaitsForCudaSemaphore));
}

void exportSemaphoresToCuda(void* cudaWaitsForVulkanSemaphoreHandle, void* vulkanWaitsForCudaSemaphoreHandle) {
    cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
    memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
    externalSemaphoreHandleDesc.flags = 0;
    externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;

    externalSemaphoreHandleDesc.handle.win32.handle = cudaWaitsForVulkanSemaphoreHandle;
    checkCudaError(cudaImportExternalSemaphore(&cudaWaitsForVulkanSemaphore, &externalSemaphoreHandleDesc));

    externalSemaphoreHandleDesc.handle.win32.handle = vulkanWaitsForCudaSemaphoreHandle;
    checkCudaError(cudaImportExternalSemaphore(&vulkanWaitsForCudaSemaphore, &externalSemaphoreHandleDesc));
}

// compresses 4 32bit floats into a 32bit uint
__device__ unsigned int rgbaFloatToInt(float4 rgba) {
    rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return ((unsigned int)(rgba.w * 255.0f) << 24) |
        ((unsigned int)(rgba.z * 255.0f) << 16) |
        ((unsigned int)(rgba.y * 255.0f) << 8) |
        ((unsigned int)(rgba.x * 255.0f));
}

__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator*(const float3& a, const float& b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float length(const float3& vec) {
    return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__device__ float3 normalize(const float3 vec) {
    float inverted_len = 1.0f / length(vec);
    return vec * inverted_len;
}

typedef struct {
    float3 o;
    float3 d;
} ray;

__device__ ray get_ray(const float& u, const float& v, float zoom, float offsetX, float offsetY) {
    ray r;
    r.o = make_float3(-3.0, 0.1, 0.1);
    r.d = normalize(make_float3(1.0 * zoom, u + offsetX / 1000, v + offsetY / 1000));
    return r;
}

__device__ float mandelbulb(float3 pos, float3 rotation) {
    float3 z = pos;

    // Rotate around X axis
    float temp_y = z.y;
    z.y = z.y * cos(rotation.x) - z.z * sin(rotation.x);
    z.z = temp_y * sin(rotation.x) + z.z * cos(rotation.x);

    // Rotate around Y axis
    float temp_x = z.x;
    z.x = z.x * cos(rotation.y) + z.z * sin(rotation.y);
    z.z = -temp_x * sin(rotation.y) + z.z * cos(rotation.y);

    // Rotate around Z axis
    temp_x = z.x;
    z.x = z.x * cos(rotation.z) - z.y * sin(rotation.z);
    z.y = temp_x * sin(rotation.z) + z.y * cos(rotation.z);

    float dr = 1.0;
    float r = 0.0;
    int iterations = 16;
    float Bailout = 4.0;
    float n = 12.0;

    for (int i = 0; i < iterations; i++) {
        r = length(z);
        if (r > Bailout) break;

        // convert to spherical coordinates
        float theta = acos(z.z / r);
        float phi = atan2(z.y, z.x);
        dr = powf(r, n - 1.0) * n * dr + 1.0;

        // scale and rotate the point
        float zr = pow(r, n);
        theta = theta * n;
        phi = phi * n;

        // convert back to cartesian coordinates
        z = make_float3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta)) * zr;

        z = z + pos;
    }
    return 0.5 * log(r) * r / dr;
}

__device__ float march(ray r, float3 rotation) {
    float total_dist = 0.0;
    int max_ray_steps = 48;
    float min_distance = 0.0005;

    int steps;
    for (steps = 0; steps < max_ray_steps; ++steps) {
        float3 p = r.o + r.d * total_dist;
        float distance = mandelbulb(p, rotation);
        total_dist += distance;
        if (distance < min_distance) break;
    }
    return 1.0 - (float)steps / (float)max_ray_steps;
}


__global__ void MandelbulbDraw(cudaSurfaceObject_t dstSurface, size_t width, size_t height, float zoom, float offsetX, float offsetY, float3 rotation)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

    float min_w_h = (float) min(width, height);

    float ar = (float)width / (float)height;
    float u = (float)x / min_w_h - ar * 0.5f;
    float v = (float)y / min_w_h - 0.5f;

    ray r = get_ray(u, v, zoom, offsetX, offsetY);
    float c = march(r, rotation);

	float4 dataOut = make_float4(c * 0.5f, c * 0.5f, c, 1.0f);

	surf2Dwrite(rgbaFloatToInt(dataOut), dstSurface, x * 4, y);
}

void renderCuda(float zoom, float offsetX, float offsetY, float t1, float t2, float t3)
{
    uint32_t nthreads = 32;
    dim3 dimGrid{ imageWidth / nthreads + 1, imageHeight / nthreads + 1};
    dim3 dimBlock{ nthreads, nthreads };
    float3 rotation = make_float3(t1, t2, t3);
    MandelbulbDraw << <dimGrid, dimBlock >> > (surfaceObject, imageWidth, imageHeight, zoom, offsetX, offsetY, rotation);
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize()); // not optimal! should be synced with vulkan using semaphores
}