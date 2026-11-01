import doctest
import hashlib
import pytest
import stringzilla as sz
_OPTIONFLAGS = doctest.NORMALIZE_WHITESPACE | doctest.IGNORE_EXCEPTION_DETAIL
def _run(module, extraglobs):
    result = doctest.testmod(module, extraglobs=extraglobs, optionflags=_OPTIONFLAGS, verbose=False)
    assert result.failed == 0, f"{result.failed} of {result.attempted} doctests failed in {module.__name__}"
    return result.attempted
def test_stringzilla_doctests():
    attempted = _run(sz, extraglobs={"sz": sz, "hashlib": hashlib})
    assert attempted > 0, "no doctests were discovered in stringzilla"
def test_stringzillas_doctests():
    try:
        import stringzillas as szs
    except ImportError as error:
        pytest.skip(f"stringzillas not importable: {error}")
    _run(szs, extraglobs={"sz": sz, "szs": szs})
