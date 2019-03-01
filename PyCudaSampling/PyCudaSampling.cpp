#define USE_PYTHON
#include "cudaCommon.h"
#include <cudaSampling.h>
#include <Python.h>
#include <numpy/arrayobject.h>

#ifndef PY_MAJOR_VERSION
#error Major version not defined!
#endif

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject* pyFuncSampling(PyObject* self, PyObject* args)
{
	PyArrayObject *DataObj;
	PyObject *pList, *tList, *sList, *fList;
	int N;

	/* Parse tuples separately since args will differ between C fcns */
	if (!PyArg_ParseTuple(args, "O!iO!O!O!O!",
		&PyArray_Type, &DataObj,
		&N,
		&PyList_Type, &pList,
		&PyList_Type, &tList,
		&PyList_Type, &sList,
		&PyList_Type, &fList
		)) {
		PyErr_SetString(PyExc_ValueError, "Parameter error!\nUsage sampling(numpy_array_of_data, N, list_of_positions, list_of_scale)parameters, list_of_flipping_parameters)");
		return NULL;
	}

	PyObject2CArray<float, float> Data(DataObj, 1);

	Py_ssize_t listSize = PyList_Size(pList);

	if(listSize != PyList_Size(tList)) {
		PyErr_SetString(PyExc_ValueError, "The lengh of theta list is inconsistent!");
		return NULL;
	}

	if(listSize != PyList_Size(sList)) {
		PyErr_SetString(PyExc_ValueError, "The lengh of scaling list is inconsistent!");
		return NULL;
	}

	if(listSize != PyList_Size(fList)) {
		PyErr_SetString(PyExc_ValueError, "The lengh of flipping list is inconsistent!");
		return NULL;
	}

	int *xBuf = (int*)malloc(listSize*sizeof(int));
	int *yBuf = (int*)malloc(listSize*sizeof(int));
	float *tBuf = (float*)malloc(listSize*sizeof(float));
	float *sBuf = (float*)malloc(listSize*sizeof(float));
	bool *fBuf = (bool*)malloc(listSize*sizeof(bool));

	PyObject *pItemList, *tItem, *sItem, *fItem;
	Py_ssize_t pItemListSize;
	PyObject *xItem, *yItem;

	for(int i = 0; i < listSize; i ++) {
		pItemList = PyList_GetItem(pList, i);

		if (!PyList_Check(pItemList)) {
			PyErr_SetString(PyExc_ValueError, "Each item in the pos list needs to be a length=2 list!");
			free(xBuf);
			free(yBuf);
			free(tBuf);
			free(sBuf);
			free(fBuf);
			return NULL;
		}

		pItemListSize = PyList_Size(pItemList);

		if (pItemListSize != 2) {
			PyErr_SetString(PyExc_TypeError, "Each item of the pos list must be two dimensions!");
			free(xBuf);
			free(yBuf);
			free(tBuf);
			free(sBuf);
			free(fBuf);
			return NULL;
		}

		xItem = PyList_GetItem(pItemList, 0);
		yItem = PyList_GetItem(pItemList, 1);

		if(!PyLong_Check(xItem) || !PyLong_Check(yItem)) {
			PyErr_SetString(PyExc_TypeError, "All x and y must be integers.");
			free(xBuf);
			free(yBuf);
			free(tBuf);
			free(sBuf);
			free(fBuf);
			return NULL;
		}

		xBuf[i] = (int)PyLong_AsLong(xItem);
		yBuf[i] = (int)PyLong_AsLong(yItem);

		tItem = PyList_GetItem(tList, i);

		if(!PyFloat_Check(tItem)) {
			PyErr_SetString(PyExc_TypeError, "All theta parameters must be floats!");
			free(xBuf);
			free(yBuf);
			free(tBuf);
			free(sBuf);
			free(fBuf);
			return NULL;
		}

		tBuf[i] = (float)PyFloat_AsDouble(tItem);

		sItem = PyList_GetItem(sList, i);

		if(!PyFloat_Check(sItem)) {
			PyErr_SetString(PyExc_TypeError, "All scale parameters must be floats!");
			free(xBuf);
			free(yBuf);
			free(tBuf);
			free(sBuf);
			free(fBuf);
			return NULL;
		}

		sBuf[i] = (float)PyFloat_AsDouble(sItem);

		fItem = PyList_GetItem(fList, i);

		if(!PyBool_Check(fItem)) {
			PyErr_SetString(PyExc_TypeError, "All flipping flags must be booleans");
			free(xBuf);
			free(yBuf);
			free(tBuf);
			free(sBuf);
			free(fBuf);
			return NULL;
		}

		fBuf[i] = PyObject_IsTrue(fItem) == 1;
	}

	float *Result = NULL;
	Result = (float*)malloc(N*N*listSize*sizeof(float));
	try {
		CALL(cudaSampling(Result, Data.getData(), Data.getColNum(), Data.getRowNum(), N, xBuf, yBuf, tBuf, sBuf, fBuf, listSize));
	}
	catch (int) {
		free(Result);
		free(xBuf);
		free(yBuf);
		free(tBuf);
		free(sBuf);
		free(fBuf);
		return NULL;
	}

	CArray2PyObject<float, float> ResultObj(Result, listSize, N*N, NPY_FLOAT, 1);

	free(Result);
	free(xBuf);
	free(yBuf);
	free(tBuf);
	free(sBuf);
	free(fBuf);

	return PyArray_Return(ResultObj.getPyArrayObject());
}

//static PyObject* pyFuncSampling(PyObject* self, PyObject* args)
//{
//	PyArrayObject *DataObj, *xyObj, *tObj, *sObj, *fObj;
//	int N;
//
//	/* Parse tuples separately since args will differ between C fcns */
//	if (!PyArg_ParseTuple(args, "O!iO!O!O!O!",
//		&PyArray_Type, &DataObj,
//		&N,
//		&PyArray_Type, &xyObj,
//		&PyArray_Type, &tObj,
//		&PyArray_Type, &sObj,
//		&PyArray_Type, &fObj
//		)) {
//		PyErr_SetString(PyExc_ValueError, "PyArg_ParseTuple error.");
//		return NULL;
//	}
//
//
//
//
//	PyObject2CArray<float, float> Data(DataObj, 1);
//	PyObject2CArray<int, int> xy(xyObj);
//	PyObject2CArray<float, float> t(tObj);
//	PyObject2CArray<float, float> s(sObj);
//	PyObject2CArray<bool, bool> f(fObj);
//
//	float *Result = NULL;
//	Result = (float*)malloc(N*N*xy.getRowNum()*sizeof(float));
//	try {
//		CALL(cudaSampling(Result, Data.getData(), Data.getColNum(), Data.getRowNum(), N, (int*)xy.getData(), (int*)((int*)xy.getData() + xy.getRowNum()), t.getData(), s.getData(), f.getData(), xy.getRowNum()));
//	}
//	catch (int) {
//		free(Result);
//		return NULL;
//	}
//
//	CArray2PyObject<float, float> ResultObj(Result, xy.getRowNum(), N*N, NPY_FLOAT, 1);
//	free(Result);
//
//	return PyArray_Return(ResultObj.getPyArrayObject());
//}





/*  define functions in module */
static PyMethodDef pyFuncMethods[] =
{
	{ "sampling", pyFuncSampling, METH_VARARGS, "sampling." },
	{ NULL, NULL, 0, NULL }
};

#if PY_MAJOR_VERSION >= 3

static int PyCudaSampling_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int PyCudaSampling_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "sampling",
        NULL,
        sizeof(struct module_state),
		pyFuncMethods,
        NULL,
		PyCudaSampling_traverse,
		PyCudaSampling_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC PyInit_PyCudaSampling(void)

#else
#define INITERROR return

/* module initialization */
PyMODINIT_FUNC initPyCudaSampling(void)
#endif
{
	import_array();

#if PY_MAJOR_VERSION >= 3
	PyObject *module = PyModule_Create(&moduledef);
#else
	PyObject *module = Py_InitModule("PyCudaSampling", pyFuncMethods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("sampling.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif

}

