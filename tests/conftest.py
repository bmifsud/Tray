import pytest
from fastapi.testclient import TestClient
from api import app, get_db
import json
import pandas as pd

class MockAsyncpgConnection:
    def __init__(self):
        self.tables = {}

    async def execute(self, query, *args):
        query = query.lower()
        if "create table" in query:
            if "ticks" in query:
                self.tables["ticks"] = []
            elif "ml_training_data" in query:
                self.tables["ml_training_data"] = []
        return "OK"

    async def copy_records_to_table(self, table_name, records, columns):
        if table_name not in self.tables:
            self.tables[table_name] = []

        for record in records:
            row_dict = dict(zip(columns, record))
            self.tables[table_name].append(row_dict)
        return "COPY"

    async def fetchval(self, query):
        if "to_regclass('public.ml_training_data')" in query:
            return "ml_training_data" if "ml_training_data" in self.tables else None
        return None

    async def fetch(self, query):
        if "from ml_training_data" in query.lower():
            return self.tables.get("ml_training_data", [])
        return []

    async def close(self):
        pass

async def override_get_db():
    # We yield the same mock connection per session for persistence between endpoints
    yield getattr(app, "mock_db_conn")

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module")
def setup_mock_db():
    app.mock_db_conn = MockAsyncpgConnection()
    yield

@pytest.fixture(scope="module")
def client(setup_mock_db):
    with TestClient(app) as c:
        yield c
