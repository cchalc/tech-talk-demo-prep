# Databricks notebook source
# MAGIC %ls /tmp

# COMMAND ----------

# MAGIC %md
# MAGIC Will use this library to test: https://github.com/keleshev/schema

# COMMAND ----------

from schema import Schema, And, Use, Optional, SchemaError

schema = Schema([{'name': And(str, len),
                  'age':  And(Use(int), lambda n: 18 <= n <= 99),
                  Optional('gender'): And(str, Use(str.lower), lambda s: s in ('squid', 'kid'))}])

data = [{'name': 'Sue', 'age': '28', 'gender': 'Squid'},
        {'name': 'Sam', 'age': '42'},
        {'name': 'Sacha', 'age': '20', 'gender': 'KID'}]


# COMMAND ----------

validated = schema.validate(data)

assert validated == [{'name': 'Sue', 'age': 28, 'gender': 'squid'},
                     {'name': 'Sam', 'age': 42},
                     {'name': 'Sacha', 'age' : 20, 'gender': 'kid'}]