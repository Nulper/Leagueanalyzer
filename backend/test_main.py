import pytest
from main import analyze_matches, get_player_data

@pytest.fixture
def client():
    from main import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_analyze_matches(client):
    response = client.post('/api/analyze-matches', json={
        'name': 'test_name',
        'tag': 'test_tag'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert 'matches' in data
    assert 'predictions' in data
    assert 'ai_response' in data

def test_get_player_data():
    api_key = "test_api_key"
    name = "test_name"
    tag = "test_tag"
    puuid = get_player_data(api_key, name, tag)
    assert puuid is not None
