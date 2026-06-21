from smart_beauty_resize import __name__ as package_name
from smart_beauty_resize import api, cli


def test_import_smoke() -> None:
    assert package_name == "smart_beauty_resize"
    assert api.__name__ == "smart_beauty_resize.api"
    assert cli.__name__ == "smart_beauty_resize.cli"
