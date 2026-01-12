try:
    import xcodec2
    print(f"xcodec2 version: {xcodec2.__version__}")
    from xcodec2.configuration_bigcodec import BigCodecConfig
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Other error: {e}")
