from warnings import catch_warnings, simplefilter
from pytest import mark
from matchms.utils import rename_deprecated_params


@mark.filterwarnings("ignore::DeprecationWarning")
def test_rename_deprecated_params():
    # pylint: disable=no-value-for-parameter
    @rename_deprecated_params({"old_param": "new_param"}, version="0.1.0")
    def example_func(new_param, another_param):
        return new_param, another_param

    # Test positional arguments
    result = example_func("some_value", "another_value")
    assert result == ("some_value", "another_value")

    # Test keyword arguments (old param)
    result = example_func(old_param="some_value", another_param="another_value")
    assert result == ("some_value", "another_value")

    # Test keyword arguments (new param)
    result = example_func(new_param="some_value", another_param="another_value")
    assert result == ("some_value", "another_value")

    # Test deprecation warnings
    with catch_warnings(record=True) as w:
        simplefilter("always")
        example_func(old_param="some_value", another_param="another_value")
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert (
            "Parameter 'old_param' is deprecated and will be removed in the future. Use 'new_param' instead. -- "
            "Deprecated since version 0.1.0"
        ) in str(w[-1].message)

    # Test if old params got removed
    @rename_deprecated_params({"old": "new"})
    def test_func(new, another):
        return new, another

    test_func(old="some_value", another="another_value")
    assert "old" not in test_func.__code__.co_varnames


def test_no_deprecated_params():
    @rename_deprecated_params({"old": "new"})
    def test_func(new, another):
        return new, another

    result = test_func(new="some_value", another="another_value")
    assert result == ("some_value", "another_value")
