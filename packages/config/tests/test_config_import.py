def test_config_package_imports() -> None:
    import internum_config

    assert internum_config.InternumBaseSettings is not None
