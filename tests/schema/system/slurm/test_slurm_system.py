from cloudai.schema.system import SlurmSystem


def test_parse_node_list_single() -> None:
    """Test parsing a list with single node names."""
    node_list = ["node1", "node2"]
    expected = ["node1", "node2"]
    assert SlurmSystem.parse_node_list(node_list) == expected


def test_parse_node_list_range() -> None:
    """Test parsing a list with a range of node names."""
    node_list = ["node[1-3]", "node5"]
    expected = ["node1", "node2", "node3", "node5"]
    assert SlurmSystem.parse_node_list(node_list) == expected


def test_parse_node_list_zero_padding() -> None:
    """Test parsing a list with zero-padded node ranges."""
    node_list = ["node[001-003]", "node005"]
    expected = ["node001", "node002", "node003", "node005"]
    assert SlurmSystem.parse_node_list(node_list) == expected
