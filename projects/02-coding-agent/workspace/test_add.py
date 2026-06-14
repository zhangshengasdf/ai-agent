"""测试文件：验证 add 函数。"""


def test_add():
    """add(2, 3) 应返回 5。"""
    from main import add

    assert add(2, 3) == 5


def test_add_negative():
    """add(-1, 1) 应返回 0。"""
    from main import add

    assert add(-1, 1) == 0


def test_add_zero():
    """add(0, 0) 应返回 0。"""
    from main import add

    assert add(0, 0) == 0


if __name__ == "__main__":
    test_add()
    test_add_negative()
    test_add_zero()
    print("All tests passed!")
