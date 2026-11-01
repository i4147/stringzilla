#include "stringzilla.h"
stringzilla_state_t *stringzilla_state_(void) {
    PyObject *module = PyState_FindModule(&stringzilla_module);
    return module ? (stringzilla_state_t *)PyModule_GetState(module) : NULL;
}
Strs **Strs_freelist_next_(Strs *node) { return (Strs **)&node->data; }
Str *Str_alloc_(void) {
    stringzilla_state_t *state = stringzilla_state_();
    if (state) sz_freelist_lock_(state);
    if (state && state->str_freelist_head) {
        Str *self = state->str_freelist_head;
        state->str_freelist_head = (Str *)self->parent; 
        state->str_freelist_count--;
        sz_freelist_unlock_(state);
        _Py_NewReference((PyObject *)self); 
        self->parent = NULL;
        self->memory.start = NULL;
        self->memory.length = 0;
        return self;
    }
    if (state) sz_freelist_unlock_(state);
    return (Str *)StrType.tp_alloc(&StrType, 0); 
}
Strs *Strs_alloc_(void) {
    stringzilla_state_t *state = stringzilla_state_();
    if (state) sz_freelist_lock_(state);
    if (state && state->strs_freelist_head) {
        Strs *self = state->strs_freelist_head;
        state->strs_freelist_head = *Strs_freelist_next_(self); 
        state->strs_freelist_count--;
        sz_freelist_unlock_(state);
        _Py_NewReference((PyObject *)self); 
        self->layout = STRS_U32_TAPE_VIEW;
        memset(&self->data, 0, sizeof(self->data));
        return self;
    }
    if (state) sz_freelist_unlock_(state);
    return (Strs *)StrsType.tp_alloc(&StrsType, 0); 
}
Strs *strs_make_empty_fragmented_(void) {
    Strs *result = Strs_alloc_();
    if (result == NULL) return (Strs *)PyErr_NoMemory();
    result->layout = STRS_FRAGMENTED;
    result->data.fragmented.count = 0;
    result->data.fragmented.spans = NULL;
    result->data.fragmented.parent = NULL;
    sz_memory_allocator_init_default(&result->data.fragmented.allocator);
    return result;
}
sz_bool_t sz_py_is_mutable(PyObject *object) {
    if (PyUnicode_Check(object)) {
        PyErr_SetString(PyExc_TypeError, "str objects are immutable (use bytearray instead)");
        return sz_false_k;
    }
    else if (PyBytes_Check(object)) {
        PyErr_SetString(PyExc_TypeError, "bytes objects are immutable (use bytearray instead)");
        return sz_false_k;
    }
    else if (PyMemoryView_Check(object)) {
        Py_buffer *view = PyMemoryView_GET_BUFFER(object);
        if (view->readonly) {
            PyErr_SetString(PyExc_TypeError, "memoryview is read-only");
            return sz_false_k;
        }
    }
    return sz_true_k;
}
int sz_py_export_optional_index(PyObject *index_obj, Py_ssize_t default_index, Py_ssize_t *result_out) {
    if (index_obj == NULL || index_obj == Py_None) {
        *result_out = default_index;
        return 1;
    }
    *result_out = PyNumber_AsSsize_t(index_obj, NULL);
    return !(*result_out == -1 && PyErr_Occurred());
}
sz_bool_t sz_py_export_string_like(PyObject *object, sz_cptr_t *start, sz_size_t *length) {
    if (PyUnicode_Check(object)) {
        Py_ssize_t signed_length;
        *start = PyUnicode_AsUTF8AndSize(object, &signed_length);
        *length = (sz_size_t)signed_length;
        return 1;
    }
    else if (PyBytes_Check(object)) {
        Py_ssize_t signed_length;
        if (PyBytes_AsStringAndSize(object, (sz_ptr_t *)start, &signed_length) == -1) {
            PyErr_SetString(PyExc_ValueError, "Couldn't access `bytes` buffer internals");
            return 0;
        }
        *length = (sz_size_t)signed_length;
        return 1;
    }
    else if (PyByteArray_Check(object)) {
        *start = PyByteArray_AS_STRING(object);
        *length = PyByteArray_GET_SIZE(object);
        return 1;
    }
    else if (PyObject_TypeCheck(object, &StrType)) {
        Str *str = (Str *)object;
        *start = str->memory.start;
        *length = str->memory.length;
        return 1;
    }
    else if (PyObject_TypeCheck(object, &FileType)) {
        File *file = (File *)object;
        *start = file->memory.start;
        *length = file->memory.length;
        return 1;
    }
    else if (PyMemoryView_Check(object)) {
        Py_buffer *view = PyMemoryView_GET_BUFFER(object);
        if (view->itemsize != 1) {
            PyErr_SetString(PyExc_ValueError, "Only single-byte integral types are supported");
            return 0;
        }
        if (!PyBuffer_IsContiguous(view, 'A')) {
            PyErr_SetString(PyExc_ValueError, "The array must be contiguous");
            return 0;
        }
        *start = (sz_cptr_t)view->buf;
        *length = (sz_size_t)view->len;
        return 1;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Unsupported argument layout");
        return 0;
    }
}
void wrap_current_exception(sz_cptr_t comment) {
    sz_unused_(comment);
}
