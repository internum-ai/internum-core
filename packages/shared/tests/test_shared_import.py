def test_shared_package_imports() -> None:
    import internum_shared

    assert internum_shared.__version__ == "0.1.0"
