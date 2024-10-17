import pyarrow.parquet as pq
table = pq.read_table('your_file.parquet')
print(table)