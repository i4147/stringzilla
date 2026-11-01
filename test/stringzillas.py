import sys
import threading
import pytest
import stringzilla as sz
import stringzillas as szs
from stringzilla import Strs
def test_library_properties():
    assert len(sz.__version__.split(".")) == 3, "Semantic versioning must be preserved"
    assert "serial" in sz.__capabilities__, "Serial backend must be present"
    assert isinstance(sz.__capabilities_str__, str) and len(sz.__capabilities_str__) > 0
    sz.reset_capabilities(sz.__capabilities__)
    assert len(szs.__version__.split(".")) == 3, "Semantic versioning must be preserved"
    assert "serial" in szs.__capabilities__, "Serial backend must be present"
    assert isinstance(szs.__capabilities_str__, str) and len(szs.__capabilities_str__) > 0
    sz.reset_capabilities(szs.__capabilities__)
def test_device_scope():
    default_scope = szs.DeviceScope()
    assert default_scope is not None
    scope_multi = szs.DeviceScope(cpu_cores=4)
    assert scope_multi is not None
    if "cuda" in szs.__capabilities__:
        try:
            scope_gpu = szs.DeviceScope(gpu_device=0)
            assert scope_gpu is not None
        except RuntimeError:
            pass
    else:
        with pytest.raises(RuntimeError):
            szs.DeviceScope(gpu_device=0)
    scope_single = szs.DeviceScope(cpu_cores=1)
    assert scope_single is not None
    scope_all = szs.DeviceScope(cpu_cores=0)
    assert scope_all is not None
    with pytest.raises(ValueError):
        szs.DeviceScope(cpu_cores=4, gpu_device=0)
    with pytest.raises(TypeError):
        szs.DeviceScope(cpu_cores="invalid")
    with pytest.raises(TypeError):
        szs.DeviceScope(gpu_device="invalid")
def test_device_scope_from_another_thread():
    scopes_and_capabilities = [(szs.DeviceScope(), None)]
    if "cuda" in szs.__capabilities__:
        try:
            scopes_and_capabilities.append((szs.DeviceScope(gpu_device=0), ("cuda",)))
        except RuntimeError:
            pass
    first = Strs(["hello", "world", "abcdef"])
    second = Strs(["hallo", "word", "abcdxf"])
    for scope, capabilities in scopes_and_capabilities:
        engine = (
            szs.LevenshteinDistances() if capabilities is None else szs.LevenshteinDistances(capabilities=capabilities)
        )
        failures = []
        def run_on_worker():
            try:
                matrix = engine(first, second, device=scope).tolist()
                diagonal = [matrix[index][index] for index in range(len(first))]
                assert diagonal == [1, 1, 1], f"unexpected distances: {diagonal}"
            except BaseException as error:
                failures.append(error)
        worker = threading.Thread(target=run_on_worker)
        worker.start()
        worker.join()
        assert not failures, f"driving a scope from another thread failed: {failures[0]!r}"
def test_cpu_scope_rejects_gpu_engine():
    if "cuda" not in szs.__capabilities__:
        pytest.skip("CUDA backend not compiled in")
    cpu_scope = szs.DeviceScope(cpu_cores=2)
    first = Strs(["hello", "world"])
    second = Strs(["hallo", "word"])
    engine = szs.LevenshteinDistances(capabilities=("cuda",))
    with pytest.raises(RuntimeError):
        engine(first, second, device=cpu_scope)
def test_to_device():
    if "cuda" not in szs.__capabilities__:
        pytest.skip("CUDA not available, skipping to_device test")
    original_strs = sz.Strs(["hello", "world", "test", "slice", "data", "", "café"])
    result = szs.to_device(original_strs)
    assert result is original_strs
    assert len(result) == 7
    assert list(result) == ["hello", "world", "test", "slice", "data", "", "café"]
    slice_strs = original_strs[1:4]
    result_slice = szs.to_device(slice_strs)
    assert result_slice is slice_strs
    assert len(result_slice) == 3
    assert list(result_slice) == ["world", "test", "slice"]
    single_strs = original_strs[2:3]
    result_single = szs.to_device(single_strs)
    assert result_single is single_strs
    assert len(result_single) == 1
    assert list(result_single) == ["test"]
    empty_strs = original_strs[3:3]
    result_empty = szs.to_device(empty_strs)
    assert result_empty is empty_strs
    assert len(result_empty) == 0
    assert list(result_empty) == []
def test_to_device_unicode_complex():
    if "cuda" not in szs.__capabilities__:
        pytest.skip("CUDA not available, skipping to_device test")
    unicode_bytes = [
        b"Hello, \xe4\xb8\x96\xe7\x95\x8c!",
        b"\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xd8\xa8\xd8\xa7\xd9\x84\xd8\xb9\xd8\xa7\xd9\x84\xd9\x85",
        b"\xf0\x9f\xa6\x96\xf0\x9f\x94\xa5\xf0\x9f\x9a\x80\xf0\x9f\x92\xbb",
        b"caf\xc3\xa9",
        b"cafe\xcc\x81",
        b"\xf0\x9d\x95\xb3\xf0\x9d\x96\x8a\xf0\x9d\x96\x91\xf0\x9d\x96\x91\xf0\x9d\x96\x94",
        b"\xe2\x80\x8d\xf0\x9f\x91\xa8\xe2\x80\x8d\xf0\x9f\x92\xbb\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x94\xac",
        b"\xe1\xbc\x88\xcf\x81\xcf\x87\xce\xb9\xce\xbc\xce\xae\xce\xb4\xce\xb7\xcf\x82",
        b"\xf0\x9f\x87\xba\xf0\x9f\x87\xb8\xf0\x9f\x87\xab\xf0\x9f\x87\xb7\xf0\x9f\x87\xaf\xf0\x9f\x87\xb5",
        b"",
        b"\xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87",
        b"\xf0\x9f\xa7\xac\xe2\x9a\x9b\xef\xb8\x8f\xf0\x9f\x94\xac",
    ]
    original_strs = sz.Strs(unicode_bytes)
    result = szs.to_device(original_strs)
    assert result is original_strs
    assert len(result) == len(unicode_bytes)
    assert list(result) == unicode_bytes
    slice_strs = original_strs[1:6]
    expected_slice = unicode_bytes[1:6]
    result_slice = szs.to_device(slice_strs)
    assert result_slice is slice_strs
    assert len(result_slice) == 5
    assert list(result_slice) == expected_slice
    for original, converted in zip(expected_slice, result_slice):
        assert bytes(converted) == original
    assert list(szs.to_device(original_strs[0:1])) == [unicode_bytes[0]]
    assert list(szs.to_device(original_strs[-1:])) == [unicode_bytes[-1]]
    assert list(szs.to_device(original_strs[9:10])) == [unicode_bytes[9]]
    nfc_nfd_slice = original_strs[3:5]
    result_forms = szs.to_device(nfc_nfd_slice)
    assert len(result_forms) == 2
    assert bytes(result_forms[0]) == unicode_bytes[3]
    assert bytes(result_forms[1]) == unicode_bytes[4]
    nested_slice = slice_strs[1:3]
    result_nested = szs.to_device(nested_slice)
    assert len(result_nested) == 2
    assert bytes(result_nested[0]) == unicode_bytes[2]
    assert bytes(result_nested[1]) == unicode_bytes[3]
def test_parameter_validation():
    with pytest.raises(TypeError):
        szs.LevenshteinDistances(open="invalid")
    with pytest.raises(TypeError):
        szs.LevenshteinDistances(extend="invalid")
    with pytest.raises(TypeError):
        szs.LevenshteinDistances(mismatch="invalid")
    with pytest.raises(TypeError):
        szs.Fingerprints(ndim="invalid")
    engine = szs.LevenshteinDistances()
    with pytest.raises((TypeError, RuntimeError)):
        engine(None, Strs(["test"]))
    rectangular = engine(Strs(["a", "b"]), Strs(["c"]))
    assert rectangular.shape == (2, 1)
    with pytest.raises((TypeError, RuntimeError)):
        engine(["test"], Strs(["test"]))
    with pytest.raises((TypeError, RuntimeError)):
        engine(Strs(["test"]), ["test"])
    fp_engine = szs.Fingerprints(ndim=5)
    with pytest.raises((TypeError, RuntimeError)):
        fp_engine(None)
    with pytest.raises((TypeError, RuntimeError)):
        fp_engine(["test"])
if __name__ == "__main__":
    sys.exit(pytest.main(["-x", "-s", __file__]))
