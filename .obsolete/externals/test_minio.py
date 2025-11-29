from unittest.mock import Mock, patch


@patch("src.externals.minio.Minio")
def test_list_buckets_returns_buckets(mock_client):
    """Test that list_buckets returns a list of buckets"""
    # Create mock bucket objects
    mock_bucket1 = Mock()
    mock_bucket1.name = "bucket1"
    mock_bucket2 = Mock()
    mock_bucket2.name = "bucket2"
    mock_bucket3 = Mock()
    mock_bucket3.name = "bucket3"

    # Configure mock to return the mock buckets
    mock_client.list_buckets.return_value = [mock_bucket1, mock_bucket2, mock_bucket3]

    # Call the method
    buckets = mock_client.list_buckets()

    # Assertions
    assert len(buckets) == 3
    assert buckets[0].name == "bucket1"
    assert buckets[1].name == "bucket2"
    assert buckets[2].name == "bucket3"


@patch("src.externals.minio.Minio")
def test_list_buckets_empty(mock_client):
    """Test list_buckets when no buckets exist"""
    # Configure mock to return empty list
    mock_client.list_buckets.return_value = []

    # Call the method
    buckets = mock_client.list_buckets()

    # Assertions
    assert len(buckets) == 0
    assert isinstance(buckets, list)


@patch("src.externals.minio.Minio")
def test_list_buckets_single_bucket(mock_client):
    """Test list_buckets with a single bucket"""
    # Create mock bucket object
    mock_bucket = Mock()
    mock_bucket.name = "my-bucket"

    # Configure mock to return single bucket
    mock_client.list_buckets.return_value = [mock_bucket]

    # Call the method
    buckets = mock_client.list_buckets()

    # Assertions
    assert len(buckets) == 1
    assert buckets[0].name == "my-bucket"
