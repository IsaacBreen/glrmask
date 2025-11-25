"""Test JSON schemas for benchmarking."""

SIMPLE_USER = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"}
    },
    "required": ["name", "age"],
    "additionalProperties": False
}

PRODUCT_ARRAY = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "number"},
            "name": {"type": "string"},
            "price": {"type": "number"}
        },
        "required": ["id", "name", "price"]
    }
}

NESTED_CONFIG = {
    "type": "object",
    "properties": {
        "server": {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "number"},
                "ssl": {"type": "boolean"}
            },
            "required": ["host", "port"]
        },
        "database": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "credentials": {
                    "type": "object",
                    "properties": {
                        "username": {"type": "string"},
                        "password": {"type": "string"}
                    },
                    "required": ["username", "password"]
                }
            },
            "required": ["url", "credentials"]
        }
    },
    "required": ["server", "database"]
}

# Test prompts for each schema
TEST_PROMPTS = {
    "simple_user": "Generate a user profile with name and age:",
    "product_array": "Create a list of products:",
    "nested_config": "Generate a server configuration:"
}

ALL_SCHEMAS = {
    "simple_user": SIMPLE_USER,
    "product_array": PRODUCT_ARRAY,
    "nested_config": NESTED_CONFIG
}
