#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#include <errno.h>

#include <cuda.h>
#include "nanoarrow/nanoarrow_device.h"
#include "nanoarrow/nanoarrow_device_cuda.h"
#include "nanoarrow/nanoarrow.h"


static struct ArrowError global_error;

class CudaTemporaryContext {
 public:
  CudaTemporaryContext(int device_id) : initialized_(false) {
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
      return;
    }
    
    err = cuDeviceGet(&device_, device_id);
    // print what cuda error is
    const char* error_string;
    cuGetErrorString(err, &error_string);
    
    if (err != CUDA_SUCCESS) {
      return;
    }
    
    err = cuDevicePrimaryCtxRetain(&context_, device_);
    if (err != CUDA_SUCCESS) {
      return;
    }

    cuCtxPushCurrent(context_);
    initialized_ = true;
  }

  bool valid() { return initialized_; }

  ~CudaTemporaryContext() {
    if (initialized_) {
      CUcontext unused;
      cuCtxPopCurrent(&unused);
      cuDevicePrimaryCtxRelease(device_);
    }
  }

 private:
  bool initialized_;
  CUdevice device_;
  CUcontext context_;
};

void pycapsule_schema_deleter(PyObject* capsule) {
  ArrowSchema* schema = (ArrowSchema*)PyCapsule_GetPointer(capsule, "arrow_schema");
  if (schema == NULL) {
    return;
  }
  ArrowFree(schema);
}

void pycapsule_array_deleter(PyObject* capsule) {
  ArrowArray* array = (ArrowArray*)PyCapsule_GetPointer(capsule, "arrow_array");
  if (array != NULL) {
    ArrowArrayRelease(array);
  }
  ArrowFree(array);
}

void pycapsule_device_array_deleter(PyObject* capsule) {
  ArrowDeviceArray* device_array = (ArrowDeviceArray*)PyCapsule_GetPointer(capsule, "arrow_device_array");
  if (device_array->array.release != NULL) {
    device_array->array.release(&device_array->array);
  }
  ArrowFree(device_array);
}

int make_simple_device_array(ArrowDeviceArray* device_array2, ArrowSchema* schema) {
  ArrowErrorSet(&global_error, "");
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowDevice* gpu = ArrowDeviceCuda(ARROW_DEVICE_CUDA, 0);
  struct ArrowArray array;
  struct ArrowDeviceArray device_array;
  struct ArrowDeviceArrayView device_array_view;
  enum ArrowType string_type = NANOARROW_TYPE_INT32;

  ArrowArrayInitFromType(&array, string_type);
  ArrowArrayStartAppending(&array);
  ArrowArrayAppendInt(&array, 1);
  ArrowArrayAppendInt(&array, 2);
  ArrowArrayAppendInt(&array, 3);
  ArrowArrayFinishBuildingDefault(&array, nullptr);

  ArrowDeviceArrayInit(cpu, &device_array, &array, nullptr);

  ArrowDeviceArrayViewInit(&device_array_view);
  ArrowArrayViewInitFromType(&device_array_view.array_view, string_type);
  ArrowDeviceArrayViewSetArray(&device_array_view, &device_array, nullptr);

  // Copy required to Cuda
  device_array2->array.release = nullptr;

  ArrowDeviceArrayViewCopy(&device_array_view, gpu, device_array2);

  // Make schema:
  ArrowSchemaInitFromType(schema, NANOARROW_TYPE_INT32);
    
  return 0;
}

static PyObject *
pyexport_make_device_array(PyObject *self, PyObject *args)
{
  struct ArrowDeviceArray* simple_device_array = (ArrowDeviceArray*)ArrowMalloc(sizeof(ArrowDeviceArray));
  struct ArrowSchema* simple_schema = (ArrowSchema*)ArrowMalloc(sizeof(ArrowSchema));
  
  simple_schema->release = NULL;

  int result = make_simple_device_array(simple_device_array, simple_schema);
  
  CudaTemporaryContext ctx(0);
  if (!ctx.valid()) {
      std::cout << "Failed to initialize CUDA context" << std::endl;
  }

  // Call alloc_c_schema(simple_schema) function
  PyObject* schema_base = PyCapsule_New(simple_schema, "arrow_schema", &pycapsule_schema_deleter);
  if (schema_base == NULL) {
    PyErr_Print();
    return NULL;  // Handle allocate() call error
  }

  PyObject* device_array_base = PyCapsule_New(simple_device_array, "arrow_device_array", &pycapsule_device_array_deleter);
  
  // Create a tuple of array and array_base objects
  PyObject *result_tuple = PyTuple_Pack(2, device_array_base, schema_base);
  
  // Cleanup references  
  Py_DECREF(schema_base);
  Py_DECREF(device_array_base);

  return result_tuple;  // Return the tuple containing array and array_base
}


static PyMethodDef PyexportMethods[] = {
  {"make_device_array",  pyexport_make_device_array, METH_VARARGS,
   "Return a simple DeviceArray and Schema."},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef pyexportmodule = {
  PyModuleDef_HEAD_INIT, "pyexport", /* name of module */
  NULL,                      /* module documentation, may be NULL */
  -1,                            /* size of per-interpreter state of the module,
                                    or -1 if the module keeps state in global variables. */
  PyexportMethods};


PyMODINIT_FUNC
PyInit_pyexport(void)
{
  return PyModule_Create(&pyexportmodule);
}
