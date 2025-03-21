from sftk.utils import flatten_list

class TestUtils:
    def test_flatten_list(self):
        """Tests the flatten_list function"""
        l = [[1, 2], [3, 4], [5, 6]]
        assert len(flatten_list(l)) == 6, "Flattened list is not the correct length"
        assert flatten_list(l) == [1, 2, 3, 4, 5, 6], "Flattened list does not contain the correct elements, or is not in the correct order"

        # Test recursive flattening
        l = [[1, 2], [3, 4], [5, [6, 7]]]
        assert len(flatten_list(l)) == 7, "Flattened list is not the correct length"
        assert flatten_list(l) == [1, 2, 3, 4, 5, 6, 7], "Flattened list does not contain the correct elements, or is not in the correct order"
