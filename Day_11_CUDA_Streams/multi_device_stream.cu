#include <stdio.h>

#if _WIN32
    //Windows threads.
#include <windows.h>

typedef HANDLE CUTThread;
typedef unsigned (WINAPI * CUT_THREADROUTINE) (void *);

#define CUT_THREADPROC unsigned WINAPI
#define  CUT_THREADEND return 0

#else
    //POSIX threads.
#include <pthread.h>

typedef pthread_t CUTThread;
typedef void *(*CUT_THREADROUTINE) (void *);

#define CUT_THREADPROC void
#define  CUT_THREADEND
#endif

//Create thread.
CUTThread start_thread (CUT_THREADROUTINE, void *data);

//Wait for thread to finish.
void end_thread (CUTThread thread);

//Destroy thread.
void destroy_thread (CUTThread thread);

//Wait for multiple threads.
void wait_for_threads (const CUTThread * threads, int num);

#if _WIN32
    //Create thread
CUTThread
start_thread (CUT_THREADROUTINE func, void *data)
{
  return CreateThread (NULL, 0, (LPTHREAD_START_ROUTINE) func, data, 0, NULL);
}

    //Wait for thread to finish
void
end_thread (CUTThread thread)
{
  WaitForSingleObject (thread, INFINITE);
  CloseHandle (thread);
}

    //Destroy thread
void
destroy_thread (CUTThread thread)
{
  TerminateThread (thread, 0);
  CloseHandle (thread);
}

    //Wait for multiple threads
void
wait_for_threads (const CUTThread * threads, int num)
{
  WaitForMultipleObjects (num, threads, true, INFINITE);

  for (int i = 0; i < num; i++)
    CloseHandle (threads[i]);
}

#else
    //Create thread
CUTThread
start_thread (CUT_THREADROUTINE func, void *data)
{
  pthread_t thread;
  pthread_create (&thread, NULL, func, data);
  return thread;
}

    //Wait for thread to finish
void
end_thread (CUTThread thread)
{
  pthread_join (thread, NULL);
}

    //Destroy thread
void
destroy_thread (CUTThread thread)
{
  pthread_cancel (thread);
}

    //Wait for multiple threads
void
wait_for_threads (const CUTThread * threads, int num)
{
  for (int i = 0; i < num; i++)
    end_thread (threads[i]);
}

#endif

#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

static void
HandleError (cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess)
    {
      printf ("%s in %s at line %d\n", cudaGetErrorString (err), file, line);
      exit (EXIT_FAILURE);
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define imin(a,b) (a<b?a:b)

#define     N    (33*1024*1024)
const int threadsPerBlock = 256;
const int blocksPerGrid =
imin (32, (N / 2 + threadsPerBlock - 1) / threadsPerBlock);

__global__ void
dot (int size, float *a, float *b, float *c)
{
  __shared__ float cache[threadsPerBlock];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  float temp = 0;
  while (tid < size)
    {
      temp += a[tid] * b[tid];
      tid += blockDim.x * gridDim.x;
    }

  // set the cache values
  cache[cacheIndex] = temp;

  // synchronize threads in this block
  __syncthreads ();

  // for reductions, threadsPerBlock must be a power of 2
  // because of the following code
  int i = blockDim.x / 2;
  while (i != 0)
    {
      if (cacheIndex < i)
	cache[cacheIndex] += cache[cacheIndex + i];
      __syncthreads ();
      i /= 2;
    }

  if (cacheIndex == 0)
    c[blockIdx.x] = cache[0];
}

struct DataStruct
{
  int deviceID;
  int size;
  float *a;
  float *b;
  float returnValue;
};

void *
routine (void *pvoidData)
{
  DataStruct *data = (DataStruct *) pvoidData;
  HANDLE_ERROR (cudaSetDevice (data->deviceID));

  int size = data->size;
  float *a, *b, c, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;

  // allocate memory on the CPU side
  a = data->a;
  b = data->b;
  partial_c = (float *) malloc (blocksPerGrid * sizeof (float));

  // allocate the memory on the GPU
  HANDLE_ERROR (cudaMalloc ((void **) &dev_a, size * sizeof (float)));
  HANDLE_ERROR (cudaMalloc ((void **) &dev_b, size * sizeof (float)));
  HANDLE_ERROR (cudaMalloc
		((void **) &dev_partial_c, blocksPerGrid * sizeof (float)));

  // copy the arrays 'a' and 'b' to the GPU
  HANDLE_ERROR (cudaMemcpy
		(dev_a, a, size * sizeof (float), cudaMemcpyHostToDevice));
  HANDLE_ERROR (cudaMemcpy
		(dev_b, b, size * sizeof (float), cudaMemcpyHostToDevice));

  dot <<< blocksPerGrid, threadsPerBlock >>> (size, dev_a, dev_b,
					      dev_partial_c);
  // copy the array 'c' back from the GPU to the CPU
  HANDLE_ERROR (cudaMemcpy
		(partial_c, dev_partial_c, blocksPerGrid * sizeof (float),
		 cudaMemcpyDeviceToHost));

  // finish up on the CPU side
  c = 0;
  for (int i = 0; i < blocksPerGrid; i++)
    {
      c += partial_c[i];
    }

  HANDLE_ERROR (cudaFree (dev_a));
  HANDLE_ERROR (cudaFree (dev_b));
  HANDLE_ERROR (cudaFree (dev_partial_c));

  // free memory on the CPU side
  free (partial_c);

  data->returnValue = c;
  return 0;
}

int
main (void)
{
  int deviceCount;
  HANDLE_ERROR (cudaGetDeviceCount (&deviceCount));
  if (deviceCount < 2)
    {
      printf ("We need at least two compute 1.0 or greater "
	      "devices, but only found %d\n", deviceCount);
      return 0;
    }

  float *a = (float *) malloc (sizeof (float) * N);
  HANDLE_NULL (a);
  float *b = (float *) malloc (sizeof (float) * N);
  HANDLE_NULL (b);

  // fill in the host memory with data
  for (int i = 0; i < N; i++)
    {
      a[i] = i;
      b[i] = i * 2;
    }

  // prepare for multithread
  DataStruct data[2];
  data[0].deviceID = 0;
  data[0].size = N / 2;
  data[0].a = a;
  data[0].b = b;

  data[1].deviceID = 1;
  data[1].size = N / 2;
  data[1].a = a + N / 2;
  data[1].b = b + N / 2;

  CUTThread thread = start_thread (routine, &(data[0]));
  routine (&(data[1]));
  end_thread (thread);

  // free memory on the CPU side
  free (a);
  free (b);

  printf ("Value calculated:  %f\n",
	  data[0].returnValue + data[1].returnValue);

  return 0;
}