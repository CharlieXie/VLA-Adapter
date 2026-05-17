class _Fnv1_32:
    def __call__(self, value):
        if isinstance(value, str):
            data = value.encode("utf-8")
        elif isinstance(value, bytes):
            data = value
        else:
            data = bytes(value)

        hval = 0x811C9DC5
        for byte in data:
            hval = (hval * 0x01000193) & 0xFFFFFFFF
            hval ^= byte
        return hval


def fnv1_32():
    return _Fnv1_32()
