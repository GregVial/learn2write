from write import get_digit


def test_get_digit():
    """Test get_digit function."""
    digit = get_digit()
    assert 0 <= digit < 10


if __name__ == "__main__":
    test_get_digit()
